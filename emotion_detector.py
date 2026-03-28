import json
import requests

WATSON_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)

HEADERS = {
    "Content-Type": "application/json",
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}

class EmotionDetectionError(Exception):
    """Custom exception for emotion detection errors."""

def emotion_detector(text):
    if not isinstance(text, str) or not text.strip():
        raise EmotionDetectionError("Input text must be a non-empty string.")

    payload = {"raw_document": {"text": text}}

    try:
        response = requests.post(WATSON_URL, headers=HEADERS, data=json.dumps(payload))
    except requests.RequestException as exc:
        raise EmotionDetectionError(f"Network error: {exc}")

    if response.status_code != 200:
        raise EmotionDetectionError(f"Watson NLP error: {response.text}")

    try:
        data = response.json()
        emotions = data["emotionPredictions"][0]["emotion"]
    except Exception:
        raise EmotionDetectionError("Unexpected response format.")

    # Determine dominant emotion
    dominant = max(emotions, key=emotions.get)

    # NEW RULE: detect invalid or meaningless text
    if all(score < 0.2 for score in emotions.values()):
        dominant = None

    # Return required output format
    return {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
        "dominant_emotion": dominant
    }