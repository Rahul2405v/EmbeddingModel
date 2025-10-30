from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

HF_API_KEY = os.getenv("HF_API_KEY")

@app.route("/embed", methods=["POST"])
def embed_text():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": text}

    response = requests.post(
        "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        return jsonify({"error": response.text}), response.status_code

    embedding = response.json()[0]
    return jsonify({"embedding": embedding, "dimensions": len(embedding)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
