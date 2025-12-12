import torch
from PIL import Image
import os
import json
import time
import cv2
import numpy as np
from diffusers import StableDiffusionXLInpaintPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor


# 設定檔案路徑
# 1. 來源圖片 (原圖)
ORIGINAL_IMAGE_PATH = r"./images/a.png"

# 2. 遮罩圖片 (來自 SAM)
MASK_IMAGE_PATH = r"./images/stage2_generated_mask.png"

# 3. 大腦決策檔案 (來自 OpenAI)
DECISION_FILE = "brain_decision.json"

# 4. 最終輸出路徑
OUTPUT_PATH = r"./images/success/stage3_final_result.png"

LORA_FOLDER = r"./models/loras"

LIBRARY_FILE = "lora_library.json"

IP_ADAPTER_PATH = r"./models/ip_adapter"
# [新增] 精修素材的路徑
REFERENCE_IMAGE_PATH = r"./images/reference_material_cutout.png"

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def composite_reference(original_pil, mask_pil, ref_pil):
    """
    1. 找出遮罩的邊界框 (Bounding Box)。
    2. 將素材縮放至該邊界框的大小。
    3. 直接貼在原圖上 (合成)。
    """
    print("正在執行「硬貼合成」 (Composite)...")
    
    # 轉 numpy 找座標
    mask_np = np.array(mask_pil.convert("L"))
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("   -> 警告: 遮罩空白，無法合成。")
        return original_pil

    # 取得遮罩範圍 (x, y, w, h)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    print(f"   -> 鎖定目標區域: x={x}, y={y}, w={w}, h={h}")
    
    # 準備素材 (確保是 RGBA 以保留去背效果)
    ref_img = ref_pil.convert("RGBA")
    
    # 縮放素材以填滿目標區域
    # 我們稍微放大一點點 (1.1倍)，確保蓋住邊緣的頭髮
    scale_up = 1.1
    new_w = int(w * scale_up)
    new_h = int(h * scale_up)
    
    resized_ref = ref_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 計算置中貼上的座標 (因為放大了，所以要往左上移一點)
    offset_x = x - (new_w - w) // 2
    offset_y = y - (new_h - h) // 2
    
    # 準備底圖
    comp_image = original_pil.convert("RGBA")
    
    # 貼上去！ (使用素材的 Alpha 通道當遮罩)
    comp_image.paste(resized_ref, (offset_x, offset_y), resized_ref)
    
    print("   -> 合成完畢！已將頭盔貼入畫面。")
    # (選用) 存出來看看貼得準不準
    # comp_image.save(ORIGINAL_IMAGE_PATH.replace(".webp", "_composite_debug.png"))
    
    return comp_image.convert("RGB")

def smart_fit_reference(ref_pil, mask_pil):
    """
    1. 測量遮罩(洞)的長寬比。
    2. 測量素材(頭盔)的實際大小。
    3. 把素材調整成跟洞一樣的比例，並居中放入正方形畫布。
    """
    if ref_pil is None or mask_pil is None: return ref_pil
    
    print("正在計算遮罩與素材的幾何關係...")
    
    # --- A. 分析遮罩 (洞的大小) ---
    mask_np = np.array(mask_pil.convert("L"))
    # 找遮罩的輪廓
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("   -> 警告: 遮罩是空的，使用預設縮放。")
        return ref_pil
    
    # 取得遮罩的邊界框 (x, y, w, h)
    largest_contour = max(contours, key=cv2.contourArea)
    mx, my, mw, mh = cv2.boundingRect(largest_contour)
    mask_ratio = mw / mh # 洞的長寬比
    print(f"   -> 洞的尺寸: {mw}x{mh} (比例: {mask_ratio:.2f})")

    # --- B. 分析素材 (頭盔的實體) ---
    # 先把 PIL 轉 numpy (保留 Alpha 通道)
    ref_np = np.array(ref_pil.convert("RGBA"))
    # 找 alpha > 0 的區域 (非透明區域)
    alpha_channel = ref_np[:, :, 3]
    coords = cv2.findNonZero(alpha_channel)
    if coords is None: return ref_pil
    rx, ry, rw, rh = cv2.boundingRect(coords)
    # 裁切出只有頭盔的部分 (去掉多餘白邊)
    helmet_crop = ref_pil.crop((rx, ry, rx+rw, ry+rh))
    print(f"   -> 頭盔實體尺寸: {rw}x{rh}")

    # --- C. 智慧合成 (Smart Padding) ---
    # 目標：創造一個正方形畫布 (IP-Adapter 喜歡正方形)，但把頭盔放進去時，
    # 要讓頭盔的「視覺比例」跟「洞的比例」一致。
    
    canvas_size = 1024 # 高畫質畫布
    canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0)) # 黑底
    
    # 計算頭盔在畫布上的目標尺寸
    # 我們希望頭盔放入畫布後，留白的方式能模仿遮罩在原圖中的感覺
    # 為了安全，我們讓頭盔佔畫布的 80% (留 20% 緩衝)，但要維持長寬比
    
    target_w = canvas_size * 0.8
    target_h = target_w / mask_ratio # 根據洞的比例反推高度
    
    # 如果推算的高度太高 (超過畫布)，就改用高度當基準
    if target_h > canvas_size * 0.8:
        target_h = canvas_size * 0.8
        target_w = target_h * mask_ratio
        
    target_w, target_h = int(target_w), int(target_h)
    
    # 縮放頭盔實體
    resized_helmet = helmet_crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    # 居中貼上
    paste_x = (canvas_size - target_w) // 2
    paste_y = (canvas_size - target_h) // 2
    
    canvas.paste(resized_helmet, (paste_x, paste_y))
    print(f"   -> 智慧適配完成！已將頭盔重塑為 {target_w}x{target_h} 以適配遮罩。")
    
    # (Debug) 存出來看看
    canvas.save(REFERENCE_IMAGE_PATH.replace(".png", "_smart_fit.png"))
    
    return canvas

# 主程式
def main():
    print("\n========== [階段 3] 啟動 SDXL Inpainting (智慧適配版) ==========")
    
    brain_data = load_json(DECISION_FILE)
    lora_library = load_json(LIBRARY_FILE)
    final_prompt = brain_data.get("final_prompt", "high quality")
    selected_key = brain_data.get("lora_key", "None")
    lora_weight = brain_data.get("lora_weight", 0.8)
    lora_info = lora_library.get(selected_key, {"filename": None})
    real_filename = lora_info.get("filename")

    print(f"大腦選擇: {selected_key}")
    # 1. 讀取基本圖片
    try:
        if not os.path.exists(ORIGINAL_IMAGE_PATH) or not os.path.exists(MASK_IMAGE_PATH):
            print("!! 錯誤: 缺圖。"); return
        image = Image.open(ORIGINAL_IMAGE_PATH).convert("RGB")
        mask_image = Image.open(MASK_IMAGE_PATH).convert("RGB")
    except Exception as e:
        print(f"圖片讀取錯誤: {e}"); return

    # 2. 處理素材 (只有在檔案存在時才執行)
    reference_image = None
    composited_image = None
    
    if os.path.exists(REFERENCE_IMAGE_PATH):
        try:
            print("發現素材，開始進行智慧適配與合成...")
            # 讀取素材
            raw_ref = Image.open(REFERENCE_IMAGE_PATH)
            
            # A. 智慧縮放 (Smart Fit)
            reference_image = smart_fit_reference(raw_ref, mask_image)
            
            # B. 強力合成 (Composite)
            composited_image = composite_reference(image, mask_image, raw_ref)
            
            # C. 遮罩擴張 (Dilation) - 有合成才需要
            mask_array = np.array(mask_image.convert("L"))
            dilation_size = 15
            kernel = np.ones((dilation_size, dilation_size), np.uint8)
            dilated_mask_array = cv2.dilate(mask_array, kernel, iterations=1)
            dilated_mask_array = cv2.GaussianBlur(dilated_mask_array, (21, 21), 0)
            mask_image = Image.fromarray(dilated_mask_array).convert("RGB")
            
        except Exception as e:
            print(f"素材處理失敗 (將降級為普通繪圖): {e}")
            reference_image = None # 重置
    else:
        print("本次不需要素材 (IP-Adapter 關閉)。")

    # 3. 載入模型 (包含 IP-Adapter 模組，備而不用)
    print("正在載入 SDXL Inpainting 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            IP_ADAPTER_PATH, subfolder="image_encoder", torch_dtype=torch.float16, use_safetensors=True
        ).to(device)

        feature_extractor = CLIPImageProcessor.from_pretrained(
            IP_ADAPTER_PATH, subfolder="image_encoder"
        )

        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor
        )
        
        # 載入權重
        pipe.load_ip_adapter(IP_ADAPTER_PATH, subfolder="", weight_name="ip-adapter_sdxl.bin")
        if device == "cuda": pipe.enable_model_cpu_offload()
        else: pipe = pipe.to(device)
        print("模型載入完成！")
        
    except Exception as e:
        print(f"模型載入失敗: {e}"); return
    
    # 4. 載入 LoRA
    use_lora = False
    if real_filename:
        lora_path = os.path.join(LORA_FOLDER, real_filename)
        print(f"正在嘗試載入 LoRA: {lora_path}") # Debug 訊息
        
        if os.path.exists(lora_path):
            try:
                pipe.load_lora_weights(lora_path)
                use_lora = True
                print(f"LoRA 掛載成功: {real_filename}")
            except Exception as e:
                # [關鍵修改] 把錯誤印出來，不要 pass
                print(f"LoRA 載入失敗！原因: {e}") 
        else:
            print(f"找不到 LoRA 檔案: {lora_path}")

    # 5. 決定參數與輸入圖
    input_image = image # 預設用原圖
    strength_value = 0.99 # 預設完全重繪
    
    if composited_image:
        strength_value = 0.65 
        print(f"\n[模式] 合成後修圖 (Composite) -> 強度 {strength_value}")
        input_image = composited_image # 用合成圖
    elif use_lora:
        strength_value = 0.75
        print(f"\n[模式] 風格轉換 (LoRA) -> 強度 {strength_value}")
    else:
        print(f"\n[模式] 一般修補 -> 強度 {strength_value}")

    print(f"正在繪製... Prompt: {final_prompt}")

    # 6. 執行生成
    # 如果原本沒有素材 (reference_image is None)，但模型已經掛載了 IP-Adapter
    # 我們必須造一張「全黑圖」來騙過模型，否則它會報錯 "requires image_embeds"
    
    if reference_image:
        final_ref_image = reference_image
        adapter_scale = 0.8  # 有素材時，強度 0.8
    else:
        # 造一張 224x224 的全黑圖
        final_ref_image = Image.new("RGB", (224, 224), (0, 0, 0))
        adapter_scale = 0.0  # [重點] 沒素材時，權重設為 0 (完全不影響生成)

    # 不要再把它塞進 cross_attention_kwargs 了，直接設定：
    pipe.set_ip_adapter_scale(adapter_scale)
    
    print(f"IP-Adapter 狀態: {'啟用' if adapter_scale > 0 else '靜音 (Scale=0)'}")

    # 處理 LoRA 的權重 (LoRA 還是要用 cross_attention_kwargs)
    cross_attention_kwargs = {}
    if use_lora: 
        cross_attention_kwargs = {"scale": lora_weight}

    result = pipe(
        prompt=final_prompt,
        negative_prompt="low quality, blurry, ugly, deformed, bad anatomy",
        image=input_image,
        mask_image=mask_image,
        ip_adapter_image=final_ref_image, # 這裡一定有圖 (真圖 或 黑圖)
        num_inference_steps=30,
        strength=strength_value,
        guidance_scale=8.0,
        cross_attention_kwargs=cross_attention_kwargs
    ).images[0]

    result.save(OUTPUT_PATH)
    print(f"完成！儲存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()