import os
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

DEFAULT_FORCED_FONT_SIZE = 50

CANDIDATE_FONTS = [
    os.path.expanduser("~/Library/Fonts/Rowdies-Bold.ttf"),

    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica.ttc",
]

def load_font(font_path: str | None, size: int = DEFAULT_FORCED_FONT_SIZE) -> ImageFont.FreeTypeFont:
    try:
        if font_path and os.path.isfile(font_path):
            return ImageFont.truetype(font_path, size)
        for fp in CANDIDATE_FONTS:
            if os.path.isfile(fp):
                return ImageFont.truetype(fp, size)

        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

def draw_bubble_text(frame_bgr: np.ndarray, text: str, xy=(40, 80),
                     text_color=(255, 182, 193),
                     bubble_color=(255, 255, 255),
                     font: ImageFont.ImageFont | None = None,
                     forced_font_size: int = DEFAULT_FORCED_FONT_SIZE) -> np.ndarray:

    if font is None:
        font = ImageFont.load_default()

    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x, y = xy

    draw.text(
        (x, y),
        text,
        font=font,
        fill=text_color,
        stroke_width=max(4, forced_font_size // 15),
        stroke_fill=bubble_color,
    )
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
