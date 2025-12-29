import torch
from PIL import Image, ImageDraw
import cv2 
import numpy as np
import os
import json
# [回歸官方標準寫法] 
from transformers import AutoProcessor, AutoModelForCausalLM 
from transformers import SamModel, SamProcessor
import time
import sys

# 如果指令是 "python run_florence... material"，代表要切素材
MODE = "main"
if len(sys.argv) > 1 and sys.argv[1] == "material":
    MODE = "material"
print(f"\n啟動偵測系統 - 模式: [{MODE.upper()}]")

# --- 1. 設定參數 ---
DECISION_FILE = "brain_decision.json"

if MODE == "material":
    # [素材模式]
    IMAGE_PATH = r"./images/reference_material.png"  # 來源是剛剛生成的素材
    OUTPUT_BOX_IMAGE_PATH = r"./images/material_box_viz.png"
    OUTPUT_MASK_IMAGE_PATH = r"./images/material_mask.png" # 素材的遮罩"
    OUTPUT_MASK_VIZ_PATH = r"./images/material_mask_visualization.png"
    OUTPUT_CUTOUT_PATH = r"./images/reference_material_cutout.png" # [重點] 切割後的成品
else:
    # [主圖模式] (預設)
    IMAGE_PATH = r"./images/a.png"
    OUTPUT_BOX_IMAGE_PATH = r"./images/stage1_found_box.png"
    OUTPUT_MASK_IMAGE_PATH = r"./images/stage2_generated_mask.png"
    OUTPUT_MASK_VIZ_PATH = r"./images/stage2_mask_visualization.png"

# --- 函數：讀取大腦決策 ---
def get_prompt_from_json():
    """
    從 brain_decision.json 讀取要偵測的文字
    優先順序：
      1. detect_prompt
      2. detect_target
      3. 預設 "object"
    """
    default = "object"
    if not os.path.exists(DECISION_FILE):
        print(f"[警告] 找不到 {DECISION_FILE}，改用預設文字: {default}")
        return default

    try:
        with open(DECISION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = data.get("detect_prompt") or data.get("detect_target") or default
        print(f"[debug] 從 {DECISION_FILE} 讀到偵測目標: {text}")
        return text
    except Exception as e:
        print(f"[警告] 讀取 {DECISION_FILE} 失敗: {e}，改用預設文字: {default}")
        return default


# --- 2. 載入圖片 ---
if not os.path.exists(IMAGE_PATH):
    print(f"!!! 錯誤: 找不到圖片 {IMAGE_PATH}")
    exit()

image_pil = Image.open(IMAGE_PATH).convert("RGB")

# 對素材模式做一個安全檢查 (避免素材生成掛了卻沒圖)
if MODE == "material":
    try:
        # 嘗試建立一張同尺寸的全白遮罩 (如果後面流程真的掛，至少不會崩潰)
        test_mask = Image.new("L", image_pil.size, 255)
    except Exception as e:
        print(f"!! 建立全白遮罩失敗: {e}")
        exit()

# --- 2. 設定模型和設備 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"正在使用設備: {device}")
start_time = time.time()

# [階段 1] 載入 Florence-2 (使用 MultimodalArt 修復版)
print("\n[階段 1] 正在載入 Florence-2-Large (Finder)...")

# [!! 關鍵 !!] 使用 multimodalart 的版本 (移除了 flash_attn)
# 這樣我們就可以放心地用 trust_remote_code=True，享受官方語法的便利
florence_model_id = "multimodalart/Florence-2-large-no-flash-attn"

florence_processor = AutoProcessor.from_pretrained(
    florence_model_id, 
    trust_remote_code=True
)

florence_model = AutoModelForCausalLM.from_pretrained(
    florence_model_id,
    torch_dtype=torch_dtype,
    trust_remote_code=True
).to(device)

# [階段 2] 載入 SAM
print("\n[階段 2] 正在載入 SAM (Hand)...")
# 使用穩定的 SAM base 模型
sam_model_id = "facebook/sam-vit-base"

SAM_AVAILABLE = True
try:
    sam_processor = SamProcessor.from_pretrained(sam_model_id)
    # ⭐ SAM 一律用 float32，避免 input(float32) vs bias(half) 不合
    sam_model = SamModel.from_pretrained(
        sam_model_id,
        torch_dtype=torch.float32
    ).to(device)
    print("  - SAM 載入成功")
except Exception as e:
    SAM_AVAILABLE = False
    sam_processor = None
    sam_model = None
    print(f"  - SAM 載入失敗，將僅使用邊界框遮罩模式。錯誤: {e}")

# ==============================================================================
# 執行階段一：Florence-2 尋找目標 (Phrase Grounding)
# ==============================================================================
# 從大腦決策檔案讀取要找什麼
TEXT_PROMPT = get_prompt_from_json()
print(f"[執行階段 1] Florence-2 正在尋找: '{TEXT_PROMPT}'...")

# 設定任務: Phrase Grounding
task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
# 官方代碼不需要手動加 <image>，Processor 會自己處理
prompt_text = task_prompt + TEXT_PROMPT

inputs = florence_processor(text=prompt_text, images=image_pil, return_tensors="pt").to(device, torch_dtype)

with torch.no_grad():
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )

# 解析結果
generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed_results = florence_processor.post_process_generation(
    generated_text, 
    task=task_prompt, 
    image_size=(image_pil.width, image_pil.height)
)

# 提取座標
if task_prompt not in parsed_results:
    print(f"  - !! 警告: 未找到目標或格式錯誤: {parsed_results}")
    # 如果沒找到，給一個預設框避免崩潰 (例如圖片中央)
    h, w = image_pil.height, image_pil.width
    boxes = [[w*0.25, h*0.25, w*0.75, h*0.75]]
    labels = ["default"]
else:
    prediction = parsed_results[task_prompt]
    boxes = prediction['bboxes']
    labels = prediction['labels']

if len(boxes) == 0:
    print("  - !! 階段 1 失敗: Florence-2 沒有找到目標。")
    exit()

# 取第一個框
best_box = boxes[0]
x1, y1, x2, y2 = best_box

print(f"  - 階段 1 成功: 找到了 '{TEXT_PROMPT}'")
print(f"    Box: {best_box}, Label: {labels[0] if labels else 'N/A'}")

# 畫出框框存成圖片
box_viz = image_pil.copy()
draw = ImageDraw.Draw(box_viz)
draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
box_viz.save(OUTPUT_BOX_IMAGE_PATH)
print(f"  - 偵測框視覺化已儲存: {OUTPUT_BOX_IMAGE_PATH}")

# ==============================================================================
# 執行階段二：SAM (Hand)
# ==============================================================================
print("\n[階段 2] 正在生成遮罩...")

# 方法 A: 繼續用 SAM (可能會切錯，只切到頭髮)
# 方法 B: 直接把 Florence-2 找到的框框塗白 (最穩，絕對包含整顆頭)

BOX_KEYWORDS = ["head", "face", "body"]

# 如果目前的搜尋目標 (TEXT_PROMPT) 文字裡包含關鍵字，就開啟強制模式
text_lower = TEXT_PROMPT.lower()
if any(k in text_lower for k in BOX_KEYWORDS):
    USE_BOX_AS_MASK = True
    print(f"偵測到目標 '{TEXT_PROMPT}' 適合使用邊界框模式。")
else:
    USE_BOX_AS_MASK = False
    print(f"偵測到目標 '{TEXT_PROMPT}' 需要精細輪廓，切換回 SAM 模式。")

if USE_BOX_AS_MASK:
    print("  - 啟用「強制邊界框模式」: 直接將偵測框轉為遮罩 (比 SAM 更穩定)。")
    
    # 建立一張全黑的圖
    mask_image = Image.new("L", image_pil.size, 0)
    draw = ImageDraw.Draw(mask_image)
    
    # 把框框區域塗白
    # 稍微把框縮小一點點 (padding)，避免邊緣太硬，或者直接用原框
    padding = 0 
    draw.rectangle([x1-padding, y1-padding, x2+padding, y2+padding], fill=255)
    
    print(f"  - 已生成矩形遮罩。")

else:
    # 這是原本的 SAM 邏輯 (如果你想切得很精細才用這個)
    print("  - 呼叫 SAM 進行精細切割...")
    input_boxes = [[best_box]] 
    sam_inputs = sam_processor(image_pil, input_boxes=input_boxes, return_tensors="pt").to(device)
    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)
    masks_tensor = sam_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(), 
        sam_inputs["original_sizes"].cpu(), 
        sam_inputs["reshaped_input_sizes"].cpu(),
        binarize=True
    )[0]
    # 選最大的
    mask_areas = [torch.sum(m).item() for m in masks_tensor]
    best_mask_idx = mask_areas.index(max(mask_areas))
    mask_numpy = masks_tensor[best_mask_idx].cpu().numpy()

    # ⭐ 保證最後是 2 維 (H, W)，避免 Pillow 出現 "Too many dimensions" 錯誤
    if mask_numpy.ndim == 3:
        # 常見幾種情況都處理一下
        if mask_numpy.shape[0] == 1:
            # (1, H, W) -> (H, W)
            mask_numpy = mask_numpy[0]
        elif mask_numpy.shape[-1] == 1:
            # (H, W, 1) -> (H, W)
            mask_numpy = mask_numpy[..., 0]
        else:
            # 例如 (C, H, W) 或 (H, W, C)，直接拿第一個 channel 當遮罩
            if mask_numpy.shape[0] <= mask_numpy.shape[-1]:
                mask_numpy = mask_numpy[0]
            else:
                mask_numpy = mask_numpy[..., 0]
    elif mask_numpy.ndim > 3:
        # 萬一維度更高，先把 size=1 的維度壓掉
        mask_numpy = np.squeeze(mask_numpy)

    if mask_numpy.ndim != 2:
        raise RuntimeError(f"Unexpected SAM mask shape: {mask_numpy.shape}")

    mask_image = Image.fromarray((mask_numpy * 255).astype(np.uint8), mode="L")

# 儲存遮罩
mask_image.save(OUTPUT_MASK_IMAGE_PATH)
print(f"  - 遮罩已儲存: {OUTPUT_MASK_IMAGE_PATH}")

# 視覺化
mask_viz = np.array(mask_image)
mask_viz_rgb = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)
mask_viz_rgb[mask_viz > 0] = [0, 255, 0] 
original_image_cv = cv2.imread(IMAGE_PATH)
viz_image = cv2.addWeighted(original_image_cv, 0.7, mask_viz_rgb, 0.3, 0)
cv2.imwrite(OUTPUT_MASK_VIZ_PATH, viz_image)

if MODE == "material":
    print("正在進行素材去背合成...")
    # 讀取原圖與剛產生的遮罩
    original = image_pil.convert("RGBA")
    mask = mask_image.convert("L")
    
    # 建立全透明底圖
    transparent_bg = Image.new("RGBA", original.size, (0, 0, 0, 0))
    
    # 利用遮罩將原圖「挖」出來，貼到透明底圖上
    cutout = Image.composite(original, transparent_bg, mask)
    
    cutout.save(OUTPUT_CUTOUT_PATH)
    print(f"完美去背素材已產出: {OUTPUT_CUTOUT_PATH}")
