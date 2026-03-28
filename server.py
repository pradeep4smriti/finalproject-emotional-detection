"""
Flask application that exposes an emotion detection endpoint.
"""

from flask import Flask, request, jsonify
from EmotionDetection import emotion_detector, EmotionDetectionError

app = Flask(__name__)


@app.route("/emotionDetector", methods=["GET"])
def emotion_detector_route():
    """
    Analyze the emotion of the provided text using the emotion_detector
    function from the EmotionDetection package.

    Returns:
        JSON response containing emotion scores and dominant emotion,
        or an error message if the input is invalid.
    """
    text_to_analyze = request.args.get("text", "")

    if not text_to_analyze:
        return (
            jsonify(
                {"error": "Please provide text to analyze using ?text=your_sentence"}
            ),
            400,
        )

    try:
        result = emotion_detector(text_to_analyze)

        if result.get("dominant_emotion") is None:
            return jsonify({"error": "Invalid text! Please try again!"}), 200

        formatted_output = {
            "anger": result["anger"],
            "disgust": result["disgust"],
            "fear": result["fear"],
            "joy": result["joy"],
            "sadness": result["sadness"],
            "dominant_emotion": result["dominant_emotion"],
        }

        return jsonify(formatted_output), 200

    except EmotionDetectionError as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
