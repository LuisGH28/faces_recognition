import cv2
from deepface import DeepFace
from .translate import EMOTIONS_ES

def analyze_emotion_from_frame(
    frame_bgr,
    detector_backend: str = "retinaface",
    enforce_detection: bool = False,
):

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    try:
        res = DeepFace.analyze(
            img_path=rgb,
            actions=["emotion"],
            enforce_detection=enforce_detection,
            detector_backend=detector_backend,
        )
        if isinstance(res, list):
            res = res[0]

        emotion_en = res.get("dominant_emotion")
        if not emotion_en:
            return {"emotion_en": None, "emotion_es": None, "score": 0.0}

        score = 0.0
        if isinstance(res.get("emotion"), dict):
            score = float(res["emotion"].get(emotion_en, 0.0))

        emotion_es = EMOTIONS_ES.get(emotion_en, emotion_en)
        return {"emotion_en": emotion_en, "emotion_es": emotion_es, "score": score}
    except Exception:
        return {"emotion_en": None, "emotion_es": None, "score": 0.0}
