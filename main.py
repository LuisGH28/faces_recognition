import sys
import platform
import cv2

from emotion import analyze_emotion_from_frame, EMOTION_COLORS
from emotion.utils import load_font, draw_bubble_text, DEFAULT_FORCED_FONT_SIZE

FONT_PATH = None

def _open_cam():
    is_mac = platform.system() == "Darwin"
    if is_mac:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def main():
    font = load_font(FONT_PATH, DEFAULT_FORCED_FONT_SIZE)
    cap = _open_cam()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        result = analyze_emotion_from_frame(frame, detector_backend="retinaface", enforce_detection=False)
        emocion_es = result.get("emotion_es")
        texto = f"Emoción: {emocion_es}" if emocion_es else "Emoción: (no detectado)"

        color = EMOTION_COLORS.get(emocion_es, (255, 182, 193))  

        frame = draw_bubble_text(
            frame_bgr=frame,
            text=texto,
            xy=(40, 80),
            text_color=color,
            bubble_color=(255, 255, 255),
            font=font,
            forced_font_size=DEFAULT_FORCED_FONT_SIZE,
        )

        cv2.imshow("Detector de emociones (kawaii grande 💖)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main())
