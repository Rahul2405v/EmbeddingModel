from flask import Flask, request, jsonify
import os
import requests

app = Flask(__name__)

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_TOKEN:
    raise ValueError("❌ Missing HF_TOKEN environment variable.")

BASE_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "✅ Hugging Face Inference Router Flask API",
        "routes": ["/embed", "/similarity"]
    }), 200


@app.route("/embed", methods=["POST"])
def embed_text():
    try:
        data = request.get_json()
        text = data.get("text")

        if not text:
            return jsonify({"error": "Missing 'text' field"}), 400

        response = requests.post(
            f"{BASE_URL}/pipeline/feature-extraction",
            headers=HEADERS,
            json={"inputs": text}
        )

        if response.status_code != 200:
            return jsonify({"error": response.text}), response.status_code

        result = response.json()
        embedding = result[0] if isinstance(result, list) and isinstance(result[0], list) else result
        return jsonify({"embedding": embedding, "dimensions": len(embedding)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/similarity", methods=["POST"])
def similarity():
    try:
        data = request.get_json()
        source_sentence = data.get("source_sentence")
        sentences = data.get("sentences")

        if not source_sentence or not sentences:
            return jsonify({"error": "Missing 'source_sentence' or 'sentences' fields"}), 400

        response = requests.post(
            f"{BASE_URL}/pipeline/sentence-similarity",
            headers=HEADERS,
            json={"inputs": {"source_sentence": source_sentence, "sentences": sentences}}
        )

        if response.status_code != 200:
            return jsonify({"error": response.text}), response.status_code

        scores = response.json()
        return jsonify({"source_sentence": source_sentence, "sentences": sentences, "similarities": scores})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

