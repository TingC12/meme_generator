import os
import json
import time

import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline

# ================= 基本路徑設定 =================

ORIGINAL_IMAGE_PATH = r"./images/a.png"
MASK_IMAGE_PATH     = r"./images/stage2_generated_mask.png"
DECISION_FILE       = "brain_decision.json"
OUTPUT_PATH         = r"./images/success/stage3_final_result.png"
LORA_LIBRARY_FILE   = "lora_library.json"

# ================= Fooocus 模型路徑 =================
FOOOCUS_CHECKPOINT_PATH = r"./models/juggernatuXL_v8_diffusers/juggernautXL_v8Rundiffusion.safetensors"
# ===================================================


def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {} if default is None else default


def load_images():
    """
    讀主圖 + 遮罩。
    如果遮罩不存在，就自動用「整張圖 = 255」當遮罩，讓整張圖都可以被重畫。
    """
    if not os.path.exists(ORIGINAL_IMAGE_PATH):
        raise FileNotFoundError(f"找不到原圖: {ORIGINAL_IMAGE_PATH}")

    image = Image.open(ORIGINAL_IMAGE_PATH).convert("RGB")

    if not os.path.exists(MASK_IMAGE_PATH):
        print(f"⚠ 找不到遮罩: {MASK_IMAGE_PATH}，改用整張圖作為遮罩")
        mask = Image.new("L", image.size, 255)
        return image, mask

    mask = Image.open(MASK_IMAGE_PATH).convert("L")
    mask = mask.point(lambda p: 255 if p > 128 else 0)
    return image, mask


def build_pipeline(device: str):
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
    如果大腦有指定 lora_key，就從 lora_library.json 下載入。
    兼容 None / 'None' / dict 格式。
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
        # 盡量從多種欄位猜路徑
        lora_path = (
            entry.get("path")
            or entry.get("file")
            or entry.get("model_path")
            or entry.get("ckpt")
            or entry.get("filename")
            or ""
        )
        # 如果只有 filename，就預設放在 ./models/lora/ 底下
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
        print(f"⚠ fuse_lora 失敗（可以先忽略，只用 load_lora_weights）: {e}")



def main():
    start_time = time.time()
    print("\n========== [第三棒] 繪圖與合成 (Fooocus SDXL) ==========")

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

    print("開始推理 (inpaint)...")
    result = pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=18,
        strength=strength,
        guidance_scale=8.0,
    ).images[0]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.save(OUTPUT_PATH)
    print(f"✅ 完成！結果已儲存到: {OUTPUT_PATH}")

    elapsed = time.time() - start_time
    print(f"本階段耗時: {elapsed:.2f} 秒")


if __name__ == "__main__":
    main()
