import json
import torch
import os
from diffusers import StableDiffusionXLPipeline

DECISION_FILE = "brain_decision.json"
MATERIAL_OUTPUT = "./images/reference_material.png"

def main():
    # 1. 讀取大腦的決策
    if not os.path.exists(DECISION_FILE):
        print("找不到決策檔案，跳過素材生成。")
        return

    with open(DECISION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 判斷是否需要執行
    if not data.get("needs_material"):
        print("大腦判斷：不需要額外素材，跳過此步驟。")
        return

    keyword = data.get("material_keyword", "object")
    print(f"\n========== [插隊任務] 素材獵人啟動 ==========")
    print(f"正在生成素材: {keyword}")

    # 3. 載入模型 (Text-to-Image) 來生成素材
    # (這裡為了示範，我們用一個標準的 SDXL 管道)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to(device)

    # 強制加入 "Close-up" (特寫) 關鍵字，確保素材只包含我們想要的局部
    prompt = f"A close-up detailed photo of {keyword}, centered, isolated on white background, cinematic lighting, photorealistic, 8k"
    image = pipe(prompt=prompt, num_inference_steps=25).images[0]
    
    image.save(MATERIAL_OUTPUT)
    print(f"素材已獲取並儲存: {MATERIAL_OUTPUT}")

if __name__ == "__main__":
    main()