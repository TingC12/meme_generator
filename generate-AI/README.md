Smart AI Image Editing Agent (智慧多模態修圖代理人)
這是一個基於 多模態代理 (Multimodal AI Agent) 架構的自動化圖像編輯系統。它不僅僅是文生圖，更是一個具備「視覺理解」與「決策能力」的 AI 系統。
它能聽懂你的中文指令，自動判斷是否需要尋找素材、自動偵測修圖區域、並透過幾何運算與光影合成技術，完成高品質的局部重繪或風格轉換。

核心特色 (Key Features):
1.智慧決策中樞 (GPT-4o Brain):
理解模糊的中文指令（如：「把頭換成鋼鐵人」）。
智慧判斷是否需要啟用「素材獵人」或掛載特定「LoRA 風格」。
內建防呆邏輯（如：換動物頭時自動加入去人體特徵咒語）。
2.強力視覺偵測 (Florence-2 + SAM):
使用微軟官方 Florence-2-large 進行精準的語意定位。
整合 SAM (Segment Anything) 進行精細切割。
Box-to-Mask 機制：針對頭部/臉部替換，強制使用邊界框遮罩，避免 SAM 只切到頭髮的常見錯誤。
3.專業級重繪 (SDXL + LoRA + IP-Adapter):
風格轉換：支援動態掛載 LoRA 模型（如：Minecraft 風、盲盒風）。
素材替換：整合 IP-Adapter，可參考特定圖片（如鋼鐵人頭盔）進行重繪。
4.幾何適配與合成 (Smart Fit & Composite):
Smart Fit: 自動計算遮罩（洞）與素材的長寬比，將素材智慧縮放至最佳比例。
Hard Composite: 先將素材透過 OpenCV 硬貼合至目標區域，再讓 AI 進行 Strength 0.65 的光影融合，徹底解決「貼圖感」與「邊緣裁切」問題。
5.系統架構 (Architecture)
系統採用模組化「接力賽」設計：
    User[用戶指令] --> Brain[階段 0: AI 大腦 (GPT-4o)]
    Brain --> Check{需要素材?}
    
    Check -- Yes --> Hunter[階段 1.5: 素材獵人]
    Hunter -->|生成去背素材| SmartFit[幾何運算 & 預先合成]
    SmartFit --> Artist
    
    Check -- No --> Detect[階段 1 & 2: 偵測與遮罩]
    Detect -->|Florence-2 + SAM| Mask[生成遮罩]
    Mask --> Artist
    
    Artist[階段 3: 藝術家 (SDXL Inpainting)]
    Artist -->|IP-Adapter / LoRA| Result[最終成品]
安裝指南 (Installation)
本專案依賴多個 AI 模型，版本相容性極為重要。請務必依照以下步驟安裝「黃金穩定版」環境。

1. 建立虛擬環境
建議使用 Anaconda：
conda create -n agent_env python=3.10
conda activate agent_env
2. 安裝 PyTorch (GPU 版)
請依照你的 CUDA 版本調整，推薦指令：
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
3. 安裝黃金組合依賴 (關鍵步驟)
pip install -r requirements.txt
4. 設定 API Key
在專案根目錄建立 .env 檔案：
OPENAI_API_KEY=sk-proj-你的OpenAI金鑰...
模型準備 (Model Setup)
請確保你的 C:\generate AI\models\ 資料夾結構如下，缺一不可：
```text
C:\generate AI\models\
│
├── loras\  (存放風格模型)
│   ├── minecraft.safetensors
│   ├── cute_blindbox_sdxl.safetensors
│   └── sdxl_cyberpunk.safetensors
│
└── ip_adapter\  (存放 IP-Adapter 模型)
    ├── ip-adapter_sdxl.bin       <-- (注意：是標準版，對應 image_encoder 的 1280 維度)
    └── image_encoder\
        ├── config.json
        ├── model.safetensors
        └── preprocessor_config.json
```
下載連結參考：
IP-Adapter: HuggingFace h94/IP-Adapter (下載 ip-adapter_sdxl.bin 及 image_encoder 資料夾內所有檔案)
LoRA: Civitai (下載 SDXL 專用 LoRA)
⚙️ 設定檔 (Configuration)
lora_library.json
請在此檔案定義你的 LoRA 模型與觸發詞。Key 必須是純英文，以便 AI 選擇。
```text
JSON
{
    "None": { "filename": null, "trigger": "" },
    "Minecraft": {
        "filename": "minecraft.safetensors",
        "trigger": "minecraft, voxel, pixel art, blocky"
    },
    "Blindbox": {
        "filename": "cute_blindbox_sdxl.safetensors",
        "trigger": "blindbox, chibi, 3d render, cute"
    }
}
```
使用方法 (Usage)

執行主程式：
更改輸入圖片的路徑後在terminal執行
python run.py
依照提示輸入指令：
換裝/換頭："幫他戴上鋼鐵人頭盔" (觸發素材獵人 + IP-Adapter + 合成重繪)
改畫風："把這張圖變成麥塊風格" (觸發 LoRA)
一般修圖："把頭髮變成紅色" (觸發一般 Inpaint)

檔案說明
run.py: 總指揮，負責串接所有階段。
run_stage0_llm.py: 大腦。負責解析意圖、選擇 LoRA、撰寫 Prompt，並決定是否需要素材。
run_stage1_5_material.py: 素材獵人。生成去背的參考素材 (如鋼鐵人頭盔)。
run_florence_plus_sam.py: 偵測與遮罩。
使用 microsoft/Florence-2-large 找座標。
針對 "head/face" 啟用 Box-to-Mask 強制矩形遮罩，確保完整覆蓋。
針對一般物體使用 SAM 進行精細切割。
run_stage3_inpaint.py: 繪圖核心。
具備 smart_fit_reference (幾何適配) 與 composite_reference (硬合成) 功能。

自動判斷模式：
有素材：強度 0.65 (融合模式)。
有 LoRA：強度 0.75 (風格模式)。
一般：強度 0.99 (重繪模式)。

