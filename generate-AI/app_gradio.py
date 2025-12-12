import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import subprocess
from datetime import datetime
import base64
import io
import json  # 👈 新增

import gradio as gr
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

from insert_word import insert_text_on_image  # 圖片嵌字核心

load_dotenv()
client = OpenAI()

# --- 路徑設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
SUCCESS_DIR = os.path.join(IMAGES_DIR, "success")
INPUT_IMAGE_PATH = os.path.join(IMAGES_DIR, "a.png")
OUTPUT_IMAGE_PATH = os.path.join(SUCCESS_DIR, "stage3_final_result.png")
RUN_PY_PATH = os.path.join(BASE_DIR, "run.py")

# 👇 新增：讀取 LoRA 清單
LORA_LIBRARY_FILE = os.path.join(BASE_DIR, "lora_library.json")
try:
    with open(LORA_LIBRARY_FILE, "r", encoding="utf-8") as f:
        _lora_lib = json.load(f)
        LORA_CHOICES = ["Auto (大腦自動)"] + list(_lora_lib.keys())
except Exception:
    # 找不到檔案就只給 Auto 選項
    LORA_CHOICES = ["Auto (大腦自動)"]


def ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(SUCCESS_DIR, exist_ok=True)


def crop_image_to_ratio(image: Image.Image, ratio_str: str) -> Image.Image:
    if not image:
        return image

    w, h = image.size
    if ratio_str == "16:9 (Landscape)":
        target_ratio = 16 / 9
    elif ratio_str == "9:16 (Portrait)":
        target_ratio = 9 / 16
    else:
        target_ratio = 1.0

    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        offset = (w - new_w) // 2
        return image.crop((offset, 0, offset + new_w, h))
    elif current_ratio < target_ratio:
        new_h = int(w / target_ratio)
        offset = (h - new_h) // 2
        return image.crop((0, offset, w, offset + new_h))
    return image


# -------------------- 文字轉圖片 (GPT) --------------------
def text2img_gpt_ui(prompt: str, ratio: str, count: int):
    prompt = (prompt or "").strip()
    if not prompt:
        prompt = "A cute baby chick working out in the gym, clean vector style."

    if ratio == "16:9 (Landscape)":
        size = "1344x768"
    elif ratio == "9:16 (Portrait)":
        size = "768x1344"
    else:
        size = "1024x1024"

    try:
        resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=max(1, int(count)),
            size=size,
        )
    except Exception as e:
        print("text2img_gpt_ui error:", e)
        return None

    if not resp.data:
        return None

    img_b64 = resp.data[0].b64_json
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    ensure_dirs()
    img.save(OUTPUT_IMAGE_PATH)
    return img


# -------------------- 圖片轉圖片 (run.py pipeline) --------------------
def run_pipeline(
    input_image: Image.Image,
    user_prompt: str,
    ratio: str,
    count: int,
    strength: float,
    lora_mode: str,   # 👈 有這個
):
    if input_image is None:
        return None, "❌ 請先上傳一張圖片"

    if not user_prompt.strip():
        user_prompt = "幫他戴個帽子"

    ensure_dirs()

    try:
        processed_image = crop_image_to_ratio(input_image, ratio)
        processed_image = processed_image.convert("RGB")
        processed_image.save(INPUT_IMAGE_PATH)
    except Exception as e:
        return None, f"❌ 圖片處理/儲存失敗: {e}"

    env = os.environ.copy()
    env["USER_DENOISING_STRENGTH"] = str(strength)

    # ✅ 這裡根據 lora_mode 設定 MANUAL_LORA_KEY
    if lora_mode and lora_mode != "Auto (大腦自動)":
        env["MANUAL_LORA_KEY"] = lora_mode
    else:
        env.pop("MANUAL_LORA_KEY", None)

    cmd = ["python", RUN_PY_PATH]

    if not os.path.exists(RUN_PY_PATH):
        return (
            processed_image,
            f"⚠️ 找不到 {RUN_PY_PATH}，只做了裁切 ({ratio})。強度: {strength}",
        )

    try:
        result = subprocess.run(
            cmd,
            input=user_prompt + "\n",
            text=True,
            capture_output=True,
            cwd=BASE_DIR,
            env=env,
        )
    except Exception as e:
        return None, f"❌ 無法執行 pipeline：{e}"

    log_text = ""
    log_text += f"=== Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
    log_text += f"=== Prompt ===\n{user_prompt}\n"
    log_text += f"=== Settings ===\nRatio: {ratio}, Strength: {strength}\n\n"
    log_text += "=== Stdout ===\n"
    log_text += result.stdout or "(No Output)\n"
    log_text += "\n=== Stderr ===\n"
    log_text += result.stderr or "(No Error)\n"

    if result.returncode != 0:
        log_text += f"\n❌ Pipeline Error Code: {result.returncode}\n"
        if os.path.exists(OUTPUT_IMAGE_PATH):
            out_img = Image.open(OUTPUT_IMAGE_PATH).convert("RGB")
            return out_img, log_text
        else:
            return None, log_text

    if not os.path.exists(OUTPUT_IMAGE_PATH):
        log_text += "\n❌ Output image not found.\n"
        return None, log_text

    out_img = Image.open(OUTPUT_IMAGE_PATH).convert("RGB")
    return out_img, log_text


def run_pipeline_ui(
    input_image: Image.Image,
    user_prompt: str,
    ratio: str,
    count: int,
    strength: float,
    lora_mode: str,   # 👈 有這個
):
    # ✅ 把 lora_mode 傳給 run_pipeline
    img, _logs = run_pipeline(input_image, user_prompt, ratio, count, strength, lora_mode)
    return img


# -------------------- 圖片嵌字 --------------------
def run_insert_word_ui(
    input_image: Image.Image,
    user_prompt: str,
    ratio: str,
    position_label: str,
):
    if input_image is None:
        return None

    ensure_dirs()

    try:
        processed_image = crop_image_to_ratio(input_image, ratio)
    except Exception:
        processed_image = input_image

    pos = "top" if position_label == "上方" else "bottom"
    img_out = insert_text_on_image(processed_image, user_prompt or "", position=pos)

    img_out.save(OUTPUT_IMAGE_PATH)
    return img_out


# -------------------- for 重新生成用的 handler --------------------
def text2img_handler(prompt, ratio, count):
    img = text2img_gpt_ui(prompt, ratio, count)
    return img, "text2img"


def img2img_handler(img, prompt, ratio, count, strength, lora_mode):
    img_out = run_pipeline_ui(img, prompt, ratio, count, strength, lora_mode)
    return img_out, "img2img"


def insert_handler(img, text, ratio, pos_label):
    img_out = run_insert_word_ui(img, text, ratio, pos_label)
    return img_out, "insert"


def regenerate(
    t2i_prompt, t2i_ratio, t2i_count,
    i2i_img, i2i_prompt, i2i_ratio, i2i_count, i2i_strength, i2i_lora_mode,
    ins_img, ins_text, ins_ratio, ins_pos,
    mode,
):
    if mode == "img2img":
        img = run_pipeline_ui(i2i_img, i2i_prompt, i2i_ratio, i2i_count, i2i_strength, i2i_lora_mode)
        return img, mode
    elif mode == "insert":
        img = run_insert_word_ui(ins_img, ins_text, ins_ratio, ins_pos)
        return img, mode
    else:
        img = text2img_gpt_ui(t2i_prompt, t2i_ratio, t2i_count)
        return img, mode


# -------------------- CSS & Layout --------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

body, .gradio-container {
    background-color: #0b0f19 !important;
    color: #e5e7eb !important;
    font-family: 'Inter', sans-serif !important;
}

/* 讓整個內容吃滿寬度，不要在中間留大空白 */
.gradio-container {
    max-width: 100% !important;
    padding: 0 20px !important;
}

footer { display: none !important; }

.sidebar-col {
    background-color: #111827 !important;
    border-right: 1px solid #1f2937;
    height: 100vh;
    padding: 20px !important;
}

.nav-bar {
    display: flex;
    flex-direction: column;
    gap: 15px;
    padding-top: 10px;
}

.nav-item {
    display: flex;
    align-items: center;
    padding: 12px 15px;
    color: #9ca3af !important;
    cursor: pointer;
    border-radius: 8px;
    transition: all 0.2s ease;
    font-size: 16px; 
    font-weight: 500;
}

.nav-item:hover, .nav-item.active {
    background-color: #3b82f6 !important; 
    color: #ffffff !important;
}

.nav-icon {
    margin-right: 12px;
    width: 24px;
    text-align: center;
    font-size: 18px;
}

.logo-area {
    font-size: 20px;
    font-weight: 700;
    color: #3b82f6 !important;
    margin-bottom: 30px;
    padding-left: 5px;
    display: flex;
    align-items: center;
    border-bottom: 1px solid #1f2937;
    padding-bottom: 20px;
}

.custom-panel {
    background-color: #111827 !important;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 24px;
}

.custom-panel .gr-block {
    background-color: transparent !important;
    border: none !important;
}

.gradio-container textarea, 
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container .gr-box {
    background-color: #1f2937 !important;
    border: 1px solid #374151 !important;
    color: #ffffff !important;
}
"""

nav_html = """
<div class="nav-bar">
    <div class="logo-area">
        <span style="font-size:24px; margin-right:8px;">🟣</span> 梗圖生成器
    </div>
    <div class="nav-item active">
        <span class="nav-icon">✨</span> 創作工具
    </div>
</div>
"""

# 👇 新增：工具介紹文字（放在 Tab 上方）
intro_markdown = """
###
**核心特色：**
- 🚀 一句中文指令完成修圖  
  例如：「把頭換成鋼鐵人機械頭」、「把這張圖改成麥塊風格」、「幫這隻狗戴上墨鏡」。

- 🧩 素材獵人模式（Material Hunter）  
  當指令需要「特定角色或機械元件」（例如鋼鐵人頭盔）時，會先自動生成乾淨去背素材，再進行幾何對齊與合成。

- 😂 梗圖創作小工具  
  - 文字 ➜ 圖片：用 GPT 幫你從文字生成梗圖。  
  - 圖片 ➜ 圖片：上傳原圖，讓 AI 幫你改背景、換頭、改畫風。  
  - 圖片嵌字：直接在圖片上疊加中文梗圖文字。

👈 **點擊左側「✨ 創作工具」開始使用！**
"""


with gr.Blocks(title="梗圖生成器") as demo:
    gr.HTML(f"<style>{custom_css}</style>")

    with gr.Row(elem_id="main_container", equal_height=True):
        # 左邊側邊欄
        with gr.Column(scale=2, min_width=200, elem_classes="sidebar-col"):
            gr.HTML(nav_html)
        
        # 右邊主工作區
        with gr.Column(scale=10):
            # 🔹 新增一個「橫向工具介紹」，佔滿右側整欄（包含生成結果上面）
            with gr.Group(elem_classes="custom-panel"):
                gr.Markdown("### 🏠 工具介紹")
                gr.Markdown(intro_markdown)

            # 底下再放「控制面板 + 生成結果」兩欄
            with gr.Row(equal_height=False):
                # 左：控制面板（約 1/3）
                with gr.Column(scale=4):
                    with gr.Group(elem_classes="custom-panel"):
                        gr.Markdown("### 🎨 梗圖生成器")
                        # ⛔ 這裡 **不要再放** intro_markdown，避免變成一整根直條
                        # gr.Markdown(intro_markdown)  ← 把原本這行刪掉

                        with gr.Tabs():
                            # 文字轉圖片
                            with gr.TabItem("文字轉圖片"):
                                text2img_prompt = gr.Textbox(
                                    label="文字轉圖片 Prompt",
                                    placeholder="描述你想要生成的畫面，例如：在健身房舉啞鈴的小雞，簡潔向量風格...",
                                    lines=6,
                                )
                                with gr.Accordion("⚙️ 進階設定 (Advanced Settings)", open=False):
                                    with gr.Row():
                                        ratio_dropdown_text2img = gr.Dropdown(
                                            ["1:1 (Square)", "16:9 (Landscape)", "9:16 (Portrait)"],
                                            label="圖片比例",
                                            value="1:1 (Square)",
                                        )
                                        count_slider_text2img = gr.Slider(
                                            1, 4, value=1, step=1,
                                            label="生成數量（目前只顯示第一張）",
                                        )
                                with gr.Row():
                                    clear_button_text2img = gr.Button("清除", variant="secondary", size="lg")
                                    run_button_text2img = gr.Button("✨ 生成圖片 (Create)", variant="primary", size="lg")

                            # 圖片轉圖片
                            with gr.TabItem("圖片轉圖片"):
                                img2img_prompt = gr.Textbox(
                                    label="提示詞 (Prompt)",
                                    placeholder="描述你想修改的內容，例如：把背景換成星空、幫人物戴上墨鏡...",
                                    lines=8,
                                )
                                gr.Markdown("#### 參考圖片 (Image Reference)")
                                img2img_input = gr.Image(
                                    label="上傳原圖",
                                    type="pil",
                                    height=240,
                                )
                                with gr.Accordion("⚙️ 進階設定 (Advanced Settings)", open=False):
                                    with gr.Row():
                                        ratio_dropdown_img2img = gr.Dropdown(
                                            ["1:1 (Square)", "16:9 (Landscape)", "9:16 (Portrait)"],
                                            label="圖片比例 (裁切)",
                                            value="1:1 (Square)",
                                        )
                                        count_slider_img2img = gr.Slider(
                                            1, 4, value=1, step=1,
                                            label="生成數量 (目前僅傳遞參數)",
                                        )
                                    strength_slider_img2img = gr.Slider(
                                        0, 1, value=0.75,
                                        label="重繪強度 (Denoising Strength)",
                                    )
                                    lora_dropdown_img2img = gr.Dropdown(
                                        choices=LORA_CHOICES,
                                        label="LoRA 風格模式",
                                        value="Auto (大腦自動)",
                                    )
                                with gr.Row():
                                    clear_button_img2img = gr.Button("清除", variant="secondary", size="lg")
                                    run_button_img2img = gr.Button("✨ 開始生成 (Create)", variant="primary", size="lg")

                            # 圖片嵌字
                            with gr.TabItem("圖片嵌字"):
                                insert_prompt = gr.Textbox(
                                    label="圖片嵌字 / 文字內容",
                                    placeholder="輸入要印在圖片上的文字，例如：今天不練，明天變廢。",
                                    lines=5,
                                )
                                gr.Markdown("#### 底圖 (Image Reference)")
                                insert_input = gr.Image(
                                    label="上傳原圖",
                                    type="pil",
                                    height=240,
                                )
                                with gr.Accordion("⚙️ 進階設定 (Advanced Settings)", open=False):
                                    ratio_dropdown_insert = gr.Dropdown(
                                        ["1:1 (Square)", "16:9 (Landscape)", "9:16 (Portrait)"],
                                        label="圖片比例 (裁切)",
                                        value="1:1 (Square)",
                                    )
                                    position_radio_insert = gr.Radio(
                                        ["下方", "上方"],
                                        label="文字位置",
                                        value="下方",
                                    )
                                with gr.Row():
                                    clear_button_insert = gr.Button("清除", variant="secondary", size="lg")
                                    run_button_insert = gr.Button("📝 嵌入文字 (Create)", variant="primary", size="lg")

                # 右：結果區（約 2/3，圖片比較大）
                with gr.Column(scale=8):
                    with gr.Group(elem_classes="custom-panel"):
                        gr.Markdown("### 🖼️ 生成結果 (Results)")

                        output_image = gr.Image(
                            label="最終效果",
                            show_label=False,
                            interactive=False,
                            height=700,
                        )

                        with gr.Row():
                            download_button = gr.DownloadButton(
                                "⬇️ 下載圖片",
                                value=OUTPUT_IMAGE_PATH,
                            )
                            regen_button = gr.Button("🔄 重新生成")

                        last_mode = gr.State("text2img")


    # 綁定事件
    run_button_text2img.click(
        fn=text2img_handler,
        inputs=[text2img_prompt, ratio_dropdown_text2img, count_slider_text2img],
        outputs=[output_image, last_mode],
    )
    clear_button_text2img.click(fn=lambda: "", inputs=None, outputs=[text2img_prompt])

    run_button_img2img.click(
        fn=img2img_handler,
        inputs=[
            img2img_input,
            img2img_prompt,
            ratio_dropdown_img2img,
            count_slider_img2img,
            strength_slider_img2img,
            lora_dropdown_img2img,  # 👈 新增            
        ],
        outputs=[output_image, last_mode],
    )
    clear_button_img2img.click(
        fn=lambda: (None, ""),
        inputs=None,
        outputs=[img2img_input, img2img_prompt],
    )

    run_button_insert.click(
        fn=insert_handler,
        inputs=[
            insert_input,
            insert_prompt,
            ratio_dropdown_insert,
            position_radio_insert,
        ],
        outputs=[output_image, last_mode],
    )
    clear_button_insert.click(
        fn=lambda: (None, ""),
        inputs=None,
        outputs=[insert_input, insert_prompt],
    )

    regen_button.click(
        fn=regenerate,
        inputs=[
            text2img_prompt, ratio_dropdown_text2img, count_slider_text2img,
            img2img_input, img2img_prompt, ratio_dropdown_img2img,
            count_slider_img2img, strength_slider_img2img, lora_dropdown_img2img,  # 👈 新增
            insert_input, insert_prompt, ratio_dropdown_insert, position_radio_insert,
            last_mode,
        ],
        outputs=[output_image, last_mode],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
