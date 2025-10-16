from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

records = []  # global in-memory store

@app.route("/upload", methods=["POST"])
def upload():
    global records
    data = request.get_json(force=True)
    if isinstance(data, list):
        records.extend(data)
        print(f"[Server] Received {len(data)} records (Total now: {len(records)})")
        return jsonify({"status": "ok", "received": len(data)}), 200
    return jsonify({"error": "Invalid format"}), 400

@app.route("/data", methods=["GET"])
def get_data():
    return jsonify(records)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
