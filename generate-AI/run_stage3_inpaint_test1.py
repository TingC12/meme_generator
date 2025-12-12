import os
import json
import time

import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline

# ----------------- 基本路徑設定 -----------------

ORIGINAL_IMAGE_PATH = r"./images/a.png"
MASK_IMAGE_PATH     = r"./images/stage2_generated_mask.png"
DECISION_FILE       = "brain_decision.json"
OUTPUT_PATH         = r"./images/success/stage3_final_result.png"
LORA_LIBRARY_FILE   = "lora_library.json"

# IP-Adapter & 素材路徑（依照 README 結構）
IP_ADAPTER_DIR          = r"./models/ip_adapter"
IP_ADAPTER_WEIGHT_PATH  = os.path.join(IP_ADAPTER_DIR, "ip-adapter_sdxl.bin")
IP_ADAPTER_ENCODER_DIR  = os.path.join(IP_ADAPTER_DIR, "image_encoder")

# 素材圖片（B 模式：只拿來當風格參考，不直接貼）
REF_MATERIAL_CUTOUT = r"./images/reference_material_cutout.png"
REF_MATERIAL_RAW    = r"./images/reference_material.png"

# Fooocus SDXL checkpoint
FOOOCUS_CHECKPOINT_PATH = r"./models/juggernatuXL_v8_diffusers/juggernautXL_v8Rundiffusion.safetensors"


# ----------------- 小工具函式 -----------------


def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {} if default is None else default


def load_images():
    """
    讀主圖 + 遮罩。
    如果遮罩不存在，就整張圖當遮罩（讓整張都可以重畫）。
    """
    if not os.path.exists(ORIGINAL_IMAGE_PATH):
        raise FileNotFoundError(f"找不到原圖: {ORIGINAL_IMAGE_PATH}")

    image = Image.open(ORIGINAL_IMAGE_PATH).convert("RGB")

    if not os.path.exists(MASK_IMAGE_PATH):
        print(f"⚠ 找不到遮罩: {MASK_IMAGE_PATH}，改用整張圖作為遮罩")
        mask = Image.new("L", image.size, 255)
        return image, mask

    mask = Image.open(MASK_IMAGE_PATH).convert("L")
    # 二值化
    mask = mask.point(lambda p: 255 if p > 128 else 0)
    return image, mask


def build_pipeline(device: str):
    """
    建立 Fooocus SDXL Inpaint pipeline
    """
    if not os.path.exists(FOOOCUS_CHECKPOINT_PATH):
        raise FileNotFoundError(f"找不到 Fooocus 模型檔: {FOOOCUS_CHECKPOINT_PATH}")

    print("載入 Fooocus SDXL checkpoint:")
    print(f"  -> {FOOOCUS_CHECKPOINT_PATH}")

    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionXLInpaintPipeline.from_single_file(
        FOOOCUS_CHECKPOINT_PATH,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("已啟用 xformers 記憶體優化")
    except Exception:
        print("未啟用 xformers（可以忽略）")

    return pipe


def maybe_apply_lora(pipe, brain: dict):
    """
    套用 LoRA（和你原本版本一樣，只是保留）
    """
    lora_key = brain.get("lora_key")
    lora_weight = float(brain.get("lora_weight", 0.8))

    # 沒指定 LoRA 的各種情況
    if (lora_key is None) or (lora_key is False):
        print("未指定 LoRA，略過 LoRA 載入 (None/False)")
        return
    if isinstance(lora_key, str) and lora_key.strip().lower() in ["", "none", "null", "no", "false"]:
        print(f"未指定 LoRA，略過 LoRA 載入 (lora_key = {lora_key})")
        return

    library = load_json(LORA_LIBRARY_FILE, {})
    if not library:
        print("⚠ lora_library.json 為空或不存在，略過 LoRA")
        return

    entry = library.get(lora_key)
    if entry is None:
        print(f"⚠ 找不到對應的 LoRA (key = {lora_key})，略過 LoRA")
        return

    # entry 可能是字串，也可能是 dict
    if isinstance(entry, dict):
        lora_path = (
            entry.get("path")
            or entry.get("file")
            or entry.get("model_path")
            or entry.get("ckpt")
            or entry.get("filename")
            or ""
        )
        if entry.get("filename") and not any(sep in lora_path for sep in ["/", "\\"]):
            lora_path = os.path.join("./models/loras", entry["filename"])
    else:
        lora_path = entry

    if not isinstance(lora_path, str) or not lora_path:
        print(f"⚠ LoRA 設定格式怪怪的 (key = {lora_key}, value = {entry})，略過 LoRA")
        return

    if not os.path.exists(lora_path):
        print(f"⚠ LoRA 檔案不存在: {lora_path}，略過 LoRA")
        return

    print(f"載入 LoRA: {lora_path} (scale={lora_weight})")
    pipe.load_lora_weights(lora_path)

    try:
        pipe.fuse_lora(lora_scale=lora_weight)
        print("已套用 LoRA fuse")
    except Exception as e:
        print(f"⚠ fuse_lora 失敗（可以忽略，只用 load_lora_weights）: {e}")


def load_style_reference():
    """
    B 模式：只把素材當「風格參考」。
    優先使用 cutout，沒有就用原素材；兩個都沒有就回傳 None。
    """
    ref_path = None
    if os.path.exists(REF_MATERIAL_CUTOUT):
        ref_path = REF_MATERIAL_CUTOUT
    elif os.path.exists(REF_MATERIAL_RAW):
        ref_path = REF_MATERIAL_RAW

    if ref_path is None:
        print("⚠ 找不到素材圖片，B 模式無法使用風格參考，改成純文字 inpaint")
        return None

    print(f"使用素材做風格參考 (style ref): {ref_path}")
    img = Image.open(ref_path).convert("RGB")
    return img


def maybe_load_ip_adapter(pipe, device: str):
    """
    嘗試載入 IP-Adapter（style 模式用）。
    若失敗就回傳 (None, False)，pipeline 照舊只用文字。
    """
    try:
        from diffusers import IPAdapterXLModel  # 需要 diffusers 新版 :contentReference[oaicite:5]{index=5}
    except Exception as e:
        print(f"⚠ 無法匯入 IPAdapterXLModel（可能 diffusers 版本太舊）：{e}")
        return None, False

    if not os.path.exists(IP_ADAPTER_WEIGHT_PATH) or not os.path.exists(IP_ADAPTER_ENCODER_DIR):
        print("⚠ 找不到 IP-Adapter 權重或 image_encoder 資料夾，略過 IP-Adapter")
        return None, False

    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        print("載入 IP-Adapter（style 模式）...")
        ip_adapter = IPAdapterXLModel.from_pretrained(
            IP_ADAPTER_DIR,
            torch_dtype=dtype,
            use_safetensors=True,
            # 如果你把 weight 放在根目錄，就不用 subfolder；失敗時可以視情況調整
            # subfolder="sdxl_models",
            weight_name=os.path.basename(IP_ADAPTER_WEIGHT_PATH),
        )

        # 把 IP-Adapter 掛到 pipeline
        pipe.load_ip_adapter(
            ip_adapter,
            image_encoder_folder=IP_ADAPTER_ENCODER_DIR,
        )

        # 設定 style 強度：0.0 = 不看圖，1.0 = 幾乎完全照圖
        pipe.set_ip_adapter_scale(0.6)  # B 模式建議 0.5~0.7
        print("IP-Adapter 載入完成，已設定 style scale = 0.6")

        return pipe, True

    except Exception as e:
        print(f"⚠ 載入 IP-Adapter 失敗，將改用純文字 inpaint: {e}")
        return None, False


# ----------------- 主程式 -----------------


def main():
    start_time = time.time()
    print("\n========== [第三棒] 繪圖與合成 (Fooocus SDXL + IP-Adapter Style) ==========")

    brain = load_json(DECISION_FILE, {})
    final_prompt = brain.get("final_prompt", "a high quality, detailed, realistic image")
    negative_prompt = brain.get("negative_prompt", "low quality, blurry, ugly, deformed, bad anatomy")

    strength_str = os.environ.get("USER_DENOISING_STRENGTH", "0.8")
    try:
        strength = float(strength_str)
    except ValueError:
        strength = 0.8

    print(f"Prompt: {final_prompt}")
    print(f"Negative prompt: {negative_prompt}")
    print(f"Denoising strength: {strength}")

    image, mask = load_images()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置: {device}")

    pipe = build_pipeline(device)
    maybe_apply_lora(pipe, brain)

    # ---- B 模式：如果大腦說「需要素材」，就嘗試載入 IP-Adapter + style ref ----
    use_style_ref = False
    style_image = None

    if brain.get("needs_material", False):
        print("大腦判斷：需要素材 → 嘗試使用 IP-Adapter style 模式（B 模式）")
        style_image = load_style_reference()
        if style_image is not None:
            pipe_with_ip, ok = maybe_load_ip_adapter(pipe, device)
            if ok:
                pipe = pipe_with_ip
                use_style_ref = True
            else:
                print("IP-Adapter 無法啟用，改用純文字 inpaint")
        else:
            print("沒有找到素材檔案，改用純文字 inpaint")
    else:
        print("大腦判斷：不需要素材 → 只用文字 inpaint")

    # ---- 開始推理 ----
    print("開始推理 (inpaint)...")

    pipe_args = dict(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=30,
        strength=strength,
        guidance_scale=8.0,
    )

    # B 模式：有成功載入 IP-Adapter 且有 style_image，就加入 image prompt
    if use_style_ref and style_image is not None:
        print("使用 IP-Adapter style 參考進行 inpaint...")
        pipe_args["ip_adapter_image"] = style_image

    result = pipe(**pipe_args).images[0]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.save(OUTPUT_PATH)
    print(f"✅ 完成！結果已儲存到: {OUTPUT_PATH}")

    elapsed = time.time() - start_time
    print(f"本階段耗時: {elapsed:.2f} 秒")


if __name__ == "__main__":
    main()
