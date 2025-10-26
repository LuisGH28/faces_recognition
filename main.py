import os
import cv2
import numpy as np
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image

FONT_PATH = "../../../../Users/usuario/Library/Fonts/Rowdies-Bold.ttf"

FORCED_FONT_SIZE = 50  

EMOTIONS_ES = {
    "happy": "feliz",
    "sad": "triste",
    "angry": "enojado",
    "fear": "asustado",
    "surprise": "sorprendido",
    "neutral": "neutral",
    "disgust": "disgustado"
}

EMOTION_COLORS = {
    "feliz": (255, 223, 128),
    "triste": (173, 216, 230),
    "enojado": (255, 160, 160),
    "asustado": (221, 160, 221),
    "sorprendido": (255, 182, 193),
    "neutral": (200, 200, 200),
    "disgustado": (144, 238, 144)
}

def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        if not os.path.isfile(font_path):
            raise OSError(f"No existe el archivo de fuente: {font_path}")
        font = ImageFont.truetype(font_path, size)
        return font
    except Exception as e:
        print(f"  No se pudo cargar la fuente TTF ({e}). "
              f"Usaré la fuente por defecto (se verá más pequeña).")
        return ImageFont.load_default()

def draw_bubble_text(frame_bgr, text, xy=(40, 80),
                     text_color=(255, 182, 193), 
                     bubble_color=(255, 255, 255),
                     font=None):
  
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x, y = xy
    pad_x, pad_y = int(FORCED_FONT_SIZE * 0.4), int(FORCED_FONT_SIZE * 0.25)
    rect_x1, rect_y1 = x - pad_x // 2, y - pad_y // 2
    rect_x2, rect_y2 = x + text_w + pad_x, y + text_h + pad_y

    draw.text(
        (x, y),
        text,
        font=font,
        fill=text_color,
        stroke_width=max(4, FORCED_FONT_SIZE // 15),
        stroke_fill=(255, 255, 255),
    )

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    font = load_font(FONT_PATH, FORCED_FONT_SIZE)

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(" No se pudo abrir la cámara. "
                           "Ve a Preferencias del Sistema > Privacidad > Cámara y permite el acceso.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        texto = "Emoción: (no detectado)"
        color_actual = (255, 182, 193)

        try:
            result = DeepFace.analyze(
                img_path=rgb,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='retinaface'
            )
            if isinstance(result, list):
                result = result[0]

            emotion = result.get('dominant_emotion')
            if emotion:
                emocion_es = EMOTIONS_ES.get(emotion, emotion)
                texto = f"Emoción: {emocion_es}"
                color_actual = EMOTION_COLORS.get(emocion_es, (255, 182, 193))
        except Exception:
            pass

        frame = draw_bubble_text(
            frame_bgr=frame,
            text=texto,
            xy=(40, 80),              
            text_color=color_actual,
            bubble_color=(255, 255, 255),
            font=font
        )

        cv2.imshow("Detector de emociones (kawaii grande 💖)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
