from flask import Flask, request, jsonify
import google.generativeai as genai
import os

app = Flask(__name__)

os.environ["GOOGLE_API_KEY"] = os.getenv("GENAI_API_KEY")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

@app.route("/api/embed", methods=["POST"])
def embed_text():
    try:
        data = request.get_json()
        text = data.get("text", None)
        if not text:
            return jsonify({"error": "Missing 'text' in request body"}), 400

        result = genai.embed_content(
            model="models/embedding-001",
            content=text
        )

        embedding = result.get("embedding", [])

        return jsonify({
            "text": text,
            "embedding": embedding,
            "dimensions": len(embedding)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
