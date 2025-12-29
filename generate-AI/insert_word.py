# insert_word.py
import os
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont


def _get_font(font_size: int) -> ImageFont.FreeTypeFont:
    """
    嘗試在不同系統上找一個可用的中文字型
    """
    candidate_paths = [
        # Windows 常見中文字型
        r"C:\Windows\Fonts\msjh.ttc",
        r"C:\Windows\Fonts\msjh.ttf",
        r"C:\Windows\Fonts\mingliu.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        # 常見西文字型（退而求其次）
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue

    # 最終保底：使用 PIL 內建字型（不一定支援中文）
    return ImageFont.load_default()


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    """
    兼容舊版 Pillow：優先用 textbbox，沒有就用 textsize
    回傳 (width, height)
    """
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h
    else:
        # 舊版 Pillow 只有 textsize
        return draw.textsize(text, font=font)


def _wrap_text_to_width(
    text: str, font: ImageFont.FreeTypeFont, max_width: int
) -> Tuple[str, int, int]:
    """
    將文字分行，讓每行寬度不超過 max_width。
    回傳 (多行文字, 整塊寬度, 整塊高度)
    """
    raw_lines = text.replace("\r", "").split("\n")

    dummy_img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy_img)

    wrapped_lines = []
    for raw in raw_lines:
        if not raw:
            wrapped_lines.append("")
            continue

        if " " in raw:
            tokens = raw.split(" ")
            joiner = " "
        else:
            tokens = list(raw)
            joiner = ""

        current = ""
        for tk in tokens:
            trial = tk if current == "" else current + joiner + tk
            w, _ = _measure_text(draw, trial, font)
            if w <= max_width:
                current = trial
            else:
                if current:
                    wrapped_lines.append(current)
                current = tk
        if current:
            wrapped_lines.append(current)

    if not wrapped_lines:
        wrapped_lines = [""]

    line_heights = []
    max_line_width = 0
    for line in wrapped_lines:
        w, h = _measure_text(draw, line, font)
        max_line_width = max(max_line_width, w)
        line_heights.append(h)

    line_spacing = int(font.size * 0.25)
    total_height = sum(line_heights) + line_spacing * (len(line_heights) - 1)

    multi_line_text = "\n".join(wrapped_lines)
    return multi_line_text, max_line_width, total_height


def insert_text_on_image(
    image: Image.Image,
    text: str,
    position: str = "bottom",
) -> Image.Image:
    """
    在圖片上疊字，回傳新的 PIL Image（不會改到原圖）。
    position: "bottom" 或 "top"
    """
    if not text.strip():
        text = "這是一張梗圖"

    img = image.convert("RGB").copy()
    w, h = img.size

    font_size = max(18, int(h * 0.07))
    font = _get_font(font_size)

    max_text_width = int(w * 0.9)
    wrapped_text, _, text_block_h = _wrap_text_to_width(
        text, font, max_text_width
    )

    draw = ImageDraw.Draw(img)

    margin = int(h * 0.03)
    if position == "top":
        text_y = margin
    else:
        text_y = h - margin - text_block_h

    stroke_width = max(2, int(font_size * 0.08))
    line_y = text_y
    for line in wrapped_text.split("\n"):
        line_w, line_h = _measure_text(draw, line, font)
        line_x = (w - line_w) // 2

        draw.text(
            (line_x, line_y),
            line,
            font=font,
            fill=(255, 255, 255),
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0),
        )
        line_y += line_h + int(font_size * 0.25)

    return img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./images/a.png")
    parser.add_argument(
        "--output", default="./images/success/stage3_final_result.png"
    )
    parser.add_argument("--text", type=str, default="")
    parser.add_argument(
        "--position", type=str, default="bottom", choices=["top", "bottom"]
    )
    args = parser.parse_args()

    if not args.text:
        args.text = input("請輸入要嵌入的文字：").strip() or "這是一張梗圖"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    img_in = Image.open(args.input).convert("RGB")
    img_out = insert_text_on_image(img_in, args.text, position=args.position)
    img_out.save(args.output)
    print(f"✅ 已輸出圖片：{args.output}")
