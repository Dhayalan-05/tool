# test_server.py
# Required packages: pip install flask flask-cors joblib scikit-learn

import os
import sqlite3
from flask import Flask, request, jsonify, abort, Response
from flask_cors import CORS
from collections import Counter
from datetime import datetime
from functools import wraps

API_KEY = None  # optional auth, None means no login required

DB_FILE = "agent_records.db"

app = Flask(__name__)
CORS(app)

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "password")  # change for security

# ---------- Database setup ----------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            browser TEXT,
            url TEXT,
            title TEXT,
            timestamp TEXT,
            system_name TEXT,
            lab_name TEXT,
            category TEXT,
            flagged INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------- Basic Auth ----------
def check_auth(username, password):
    return username == ADMIN_USER and password == ADMIN_PASS

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response('Login required', 401, {'WWW-Authenticate':'Basic realm="Login Required"'})
        return f(*args, **kwargs)
    return decorated

# ---------- Helpers ----------
def insert_records(records):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    for r in records:
        cur.execute("""
            INSERT INTO records (browser,url,title,timestamp,system_name,lab_name,category,flagged)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            r.get("browser",""),
            r.get("url",""),
            r.get("title",""),
            r.get("timestamp",""),
            r.get("system_name",""),
            r.get("lab_name",""),
            r.get("category",""),
            r.get("flagged",0)
        ))
    conn.commit()
    conn.close()

def fetch_records(date=None):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    query = "SELECT id,browser,url,title,timestamp,system_name,lab_name,category,flagged FROM records"
    params = []
    if date:
        query += " WHERE date(timestamp)=?"
        params.append(date)
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    data = []
    for r in rows:
        data.append({
            "id": r[0],
            "browser": r[1],
            "url": r[2],
            "title": r[3],
            "timestamp": r[4],
            "system_name": r[5],
            "lab_name": r[6],
            "category": r[7],
            "flagged": r[8]
        })
    return data

def update_flag(record_id, flagged):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("UPDATE records SET flagged=? WHERE id=?", (flagged, record_id))
    conn.commit()
    conn.close()

# ---------- API Endpoints ----------
@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error":"no data"}), 400
    if isinstance(data, list):
        insert_records(data)
        return jsonify({"status":"ok","received":len(data)}), 200
    elif isinstance(data, dict):
        insert_records([data])
        return jsonify({"status":"ok","received":1}), 200
    else:
        return jsonify({"error":"bad format"}), 400

@app.route("/labs", methods=["GET"])
@requires_auth
def get_labs():
    date = request.args.get("date")
    all_records = fetch_records(date)
    lab_map = {}
    for r in all_records:
        lab = r.get("lab_name","Unknown Lab")
        sys = r.get("system_name","Unknown System")
        flag = r.get("flagged",0)
        lab_map.setdefault(lab, {}).setdefault(sys, 0)
        lab_map[lab][sys] = max(lab_map[lab][sys], flag)
    output = {}
    for lab, systems in lab_map.items():
        output[lab] = [{"system_name":s,"flagged":systems[s]} for s in systems]
    return jsonify(output),200

@app.route("/lab_summary", methods=["GET"])
@requires_auth
def lab_summary():
    lab = request.args.get("lab")
    date = request.args.get("date")
    if not lab:
        return jsonify({"error":"missing lab param"}),400
    records = fetch_records(date)
    cnt = Counter()
    for r in records:
        if r.get("lab_name") == lab:
            cnt[r.get("category","Other")] += 1
    return jsonify(dict(cnt)),200

@app.route("/system_summary", methods=["GET"])
@requires_auth
def system_summary():
    system = request.args.get("system")
    date = request.args.get("date")
    if not system:
        return jsonify({"error":"missing system param"}),400
    records = fetch_records(date)
    cnt = Counter()
    for r in records:
        if r.get("system_name") == system:
            cnt[r.get("category","Other")] += 1
    return jsonify(dict(cnt)),200

@app.route("/data", methods=["GET"])
@requires_auth
def get_data():
    date = request.args.get("date")
    records = fetch_records(date)
    raw_data = []
    for r in records[-50:]:
        raw_data.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "system_name": r["system_name"],
            "browser": r["browser"],
            "title": r["title"],
            "url": r["url"],
            "category": r["category"],
            "flagged": r["flagged"]
        })
    return jsonify(raw_data),200

@app.route("/unflag", methods=["POST"])
@requires_auth
def unflag_entry():
    req = request.get_json(force=True)
    record_id = req.get("id")
    if record_id is None:
        return jsonify({"error":"missing id"}),400
    update_flag(record_id, 0)
    print(f"[Server] Record {record_id} unflagged. Retraining ML...")
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier

        records = fetch_records()
        texts = [r["title"] + " " + r["url"] for r in records if r["flagged"]==0]
        labels = [r["category"] for r in records if r["flagged"]==0]

        if texts and labels:
            vectorizer = TfidfVectorizer(ngram_range=(1,2))
            X = vectorizer.fit_transform(texts)
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X, labels)
            joblib.dump(vectorizer, "vectorizer.pkl")
            joblib.dump(clf, "category_model.pkl")
            print(f"[ML] Retrained model on {len(texts)} unflagged records.")
        else:
            print("[ML] Not enough unflagged records to retrain.")
    except Exception as e:
        print("[ML] Retrain failed:", e)
    return jsonify({"status":"ok","id":record_id}),200

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000)
