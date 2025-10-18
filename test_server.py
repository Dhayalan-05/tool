# test_server.py
# Required packages:
# pip install flask flask-cors

from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from collections import defaultdict, Counter

app = Flask(__name__)
CORS(app)

records = []  # in-memory storage

API_KEY = None  # optional auth

@app.route("/upload", methods=["POST"])
def upload():
    if API_KEY:
        key = request.headers.get("X-API-KEY")
        if key != API_KEY:
            abort(401)
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "no data"}), 400
    if isinstance(data, list):
        records.extend(data)
        print(f"[Server] Received {len(data)} records (total: {len(records)})")
        return jsonify({"status":"ok","received":len(data)}), 200
    elif isinstance(data, dict):
        records.append(data)
        print(f"[Server] Received 1 record (total: {len(records)})")
        return jsonify({"status":"ok","received":1}), 200
    else:
        return jsonify({"error":"bad format"}), 400

@app.route("/labs", methods=["GET"])
def get_labs():
    lab_map = defaultdict(set)
    for r in records:
        lab_map[r.get("lab_name","Unknown Lab")].add(r.get("system_name","Unknown System"))
    return jsonify({lab: list(systems) for lab, systems in lab_map.items()}), 200

@app.route("/lab_summary", methods=["GET"])
def lab_summary():
    lab = request.args.get("lab")
    if not lab:
        return jsonify({"error":"missing lab param"}), 400
    cnt = Counter()
    for r in records:
        if r.get("lab_name") == lab:
            cnt[r.get("category","Other")] += 1
    return jsonify(dict(cnt)), 200

@app.route("/system_summary", methods=["GET"])
def system_summary():
    system = request.args.get("system")
    if not system:
        return jsonify({"error":"missing system param"}), 400
    cnt = Counter()
    for r in records:
        if r.get("system_name") == system:
            cnt[r.get("category","Other")] += 1
    return jsonify(dict(cnt)), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
