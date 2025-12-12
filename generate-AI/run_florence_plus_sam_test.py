import torch
from PIL import Image, ImageDraw
import cv2
import numpy as np
import os
import json
import time
import sys

# [回歸官方標準寫法] 
from transformers import AutoProcessor, AutoModelForCausalLM 
from transformers import SamModel, SamProcessor

# 如果指令是 "python run_florence_plus_sam.py material"，代表要切素材
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
    OUTPUT_MASK_IMAGE_PATH = r"./images/material_mask.png"  # 素材的遮罩
    OUTPUT_MASK_VIZ_PATH = r"./images/material_mask_visualization.png"
    OUTPUT_CUTOUT_PATH = r"./images/reference_material_cutout.png"  # [重點] 切割後的成品
else:
    # [主圖模式] (預設)
    IMAGE_PATH = r"./images/a.png"
    OUTPUT_BOX_IMAGE_PATH = r"./images/stage1_found_box.png"
    OUTPUT_MASK_IMAGE_PATH = r"./images/stage2_generated_mask.png"
    OUTPUT_MASK_VIZ_PATH = r"./images/stage2_mask_visualization.png"


# --- 函數：讀取大腦決策 ---
def get_prompt_from_json():
    """
    從 brain_decision.json 裡拿要偵測的目標文字。
    優先用 detect_prompt，沒有的話再 fallback。
    """
    default = "object"
    if not os.path.exists(DECISION_FILE):
        return default
    try:
        with open(DECISION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 正確的 key 是 detect_prompt
        return data.get("detect_prompt") or data.get("detect_target") or default
    except Exception as e:
        print(f"[警告] 讀取 {DECISION_FILE} 失敗: {e}")
        return default


# --- 2. 載入圖片 ---
if not os.path.exists(IMAGE_PATH):
    print(f"!!! 錯誤: 找不到圖片 {IMAGE_PATH}")
    sys.exit(1)

image_pil = Image.open(IMAGE_PATH).convert("RGB")

# 對素材模式做一個安全檢查 (避免素材生成掛了卻沒圖)
if MODE == "material":
    try:
        # 嘗試建立一張同尺寸的全白遮罩 (如果後面流程真的掛，至少不會崩潰)
        test_mask = Image.new("L", image_pil.size, 255)
    except Exception as e:
        print(f"!! 建立全白遮罩失敗: {e}")
        sys.exit(1)


# --- 2. 設定模型和設備 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"正在使用設備: {device}")
start_time = time.time()

# [階段 1] 載入 Florence-2 (使用 MultimodalArt 修復版)
print("\n[階段 1] 正在載入 Florence-2-Large (Finder)...")

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

# [階段 2] 嘗試載入 SAM
print("\n[階段 2] 正在載入 SAM (Hand)...")
# 修正：使用存在的官方 checkpoint
sam_model_id = "facebook/sam-vit-base"

SAM_AVAILABLE = True
try:
    sam_processor = SamProcessor.from_pretrained(sam_model_id)
    sam_model = SamModel.from_pretrained(
        sam_model_id,
        torch_dtype=torch_dtype
    ).to(device)
    print("  - SAM 載入成功")
except Exception as e:
    SAM_AVAILABLE = False
    sam_processor = None
    sam_model = None
    print(f"  - SAM 載入失敗，將僅使用方框遮罩模式。錯誤訊息：{e}")

print(f"  - 所有必要模型載入完畢 (耗時: {time.time()-start_time:.2f} 秒)\n")


# ==============================================================================
# 執行階段一：Florence-2 尋找目標 (Phrase Grounding)
# ==============================================================================
TEXT_PROMPT = get_prompt_from_json()
print(f"[執行階段 1] Florence-2 正在尋找: '{TEXT_PROMPT}'...")

task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
prompt_text = task_prompt + TEXT_PROMPT

inputs = florence_processor(
    text=prompt_text,
    images=image_pil,
    return_tensors="pt"
).to(device, torch_dtype)

with torch.no_grad():
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False,
    )

generated_text = florence_processor.batch_decode(
    generated_ids, skip_special_tokens=False
)[0]

parsed_results = florence_processor.post_process_generation(
    generated_text,
    task=task_prompt,
    image_size=(image_pil.width, image_pil.height),
)

if task_prompt not in parsed_results:
    print(f"  - !! 警告: 未找到目標或格式錯誤: {parsed_results}")
    # 如果沒找到，給一個預設框避免崩潰 (例如圖片中央)
    h, w = image_pil.height, image_pil.width
    boxes = [[w * 0.25, h * 0.25, w * 0.75, h * 0.75]]
    labels = ["default"]
else:
    prediction = parsed_results[task_prompt]
    boxes = prediction["bboxes"]
    labels = prediction["labels"]

if len(boxes) == 0:
    print("  - !! 階段 1 失敗: Florence-2 沒有找到目標。")
    sys.exit(1)

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
# ⭐ 特例：WHOLE_IMAGE 直接用整張圖當遮罩，不使用 SAM
# ==============================================================================
if TEXT_PROMPT.strip().upper() == "WHOLE_IMAGE":
    print("\n[階段 2] WHOLE_IMAGE 模式：直接使用整張圖作為遮罩（不呼叫 SAM）")

    # 全白遮罩
    mask_image = Image.new("L", image_pil.size, 255)
    mask_image.save(OUTPUT_MASK_IMAGE_PATH)
    print(f"  - WHOLE_IMAGE 遮罩已儲存: {OUTPUT_MASK_IMAGE_PATH}")

    # 視覺化（整張紅色半透明疊加）
    w, h = image_pil.size
    viz = image_pil.convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (255, 0, 0, 80))
    viz = Image.alpha_composite(viz, overlay).convert("RGB")
    viz.save(OUTPUT_MASK_VIZ_PATH)
    print(f"  - WHOLE_IMAGE 遮罩視覺化已儲存: {OUTPUT_MASK_VIZ_PATH}")

    # 如果是素材模式，順便去背一次
    if MODE == "material":
        print("正在進行素材去背合成 (WHOLE_IMAGE)...")
        original = image_pil.convert("RGBA")
        mask = mask_image.convert("L")
        transparent_bg = Image.new("RGBA", original.size, (0, 0, 0, 0))
        cutout = Image.composite(original, transparent_bg, mask)
        cutout.save(OUTPUT_CUTOUT_PATH)
        print(f"完美去背素材已產出: {OUTPUT_CUTOUT_PATH}")

    # 完成後直接結束程式
    sys.exit(0)


# ==============================================================================
# 執行階段二：SAM / Box-to-Mask（一般情況）
# ==============================================================================
print("\n[階段 2] 正在生成遮罩...")

BOX_KEYWORDS = ["head", "face", "body"]

text_lower = TEXT_PROMPT.lower()
if any(k in text_lower for k in BOX_KEYWORDS):
    USE_BOX_AS_MASK = True
    print(f"偵測到目標 '{TEXT_PROMPT}' 適合使用邊界框模式。")
else:
    USE_BOX_AS_MASK = False
    print(f"偵測到目標 '{TEXT_PROMPT}' 需要精細輪廓，預計使用 SAM。")

# --- 如果 SAM 不能用，強制改用方框模式 ---
if not SAM_AVAILABLE:
    if not USE_BOX_AS_MASK:
        print("  - SAM 不可用，將從精細模式降級為『邊界框遮罩模式』。")
    USE_BOX_AS_MASK = True

if USE_BOX_AS_MASK:
    print("  - 啟用「強制邊界框模式」: 直接將偵測框轉為遮罩。")
    mask_image = Image.new("L", image_pil.size, 0)
    draw = ImageDraw.Draw(mask_image)

    padding = 0
    draw.rectangle(
        [x1 - padding, y1 - padding, x2 + padding, y2 + padding],
        fill=255,
    )
    print("  - 已生成矩形遮罩。")

else:
    # SAM 精細切割模式
    print("  - 呼叫 SAM 進行精細切割...")
    input_boxes = [[best_box]]
    sam_inputs = sam_processor(
        image=image_pil, input_boxes=input_boxes, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    masks_tensor = sam_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu(),
        binarize=True,
    )[0]

    mask_areas = [torch.sum(m).item() for m in masks_tensor]
    best_mask_idx = mask_areas.index(max(mask_areas))
    mask_numpy = masks_tensor[best_mask_idx].cpu().numpy()

    # ⭐ 保證最後是 2 維 (H, W)，避免 Pillow 出現 "Too many dimensions" 錯誤
    if mask_numpy.ndim == 3:
        if mask_numpy.shape[0] == 1:
            mask_numpy = mask_numpy[0]
        elif mask_numpy.shape[-1] == 1:
            mask_numpy = mask_numpy[..., 0]
        else:
            if mask_numpy.shape[0] <= mask_numpy.shape[-1]:
                mask_numpy = mask_numpy[0]
            else:
                mask_numpy = mask_numpy[..., 0]
    elif mask_numpy.ndim > 3:
        mask_numpy = np.squeeze(mask_numpy)

    if mask_numpy.ndim != 2:
        raise RuntimeError(f"Unexpected SAM mask shape: {mask_numpy.shape}")

    mask_image = Image.fromarray((mask_numpy * 255).astype(np.uint8), mode="L")

# --- 儲存遮罩 ---
mask_image.save(OUTPUT_MASK_IMAGE_PATH)
print(f"  - 遮罩已儲存: {OUTPUT_MASK_IMAGE_PATH}")

# 視覺化
mask_viz = np.array(mask_image)
mask_viz_rgb = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)
mask_viz_rgb[mask_viz > 0] = [0, 255, 0]
original_image_cv = cv2.imread(IMAGE_PATH)
viz_image = cv2.addWeighted(original_image_cv, 0.7, mask_viz_rgb, 0.3, 0)
cv2.imwrite(OUTPUT_MASK_VIZ_PATH, viz_image)
print(f"  - 遮罩視覺化已儲存: {OUTPUT_MASK_VIZ_PATH}")

if MODE == "material":
    print("正在進行素材去背合成...")
    original = image_pil.convert("RGBA")
    mask = mask_image.convert("L")

    transparent_bg = Image.new("RGBA", original.size, (0, 0, 0, 0))
    cutout = Image.composite(original, transparent_bg, mask)
    cutout.save(OUTPUT_CUTOUT_PATH)
    print(f"完美去背素材已產出: {OUTPUT_CUTOUT_PATH}")
