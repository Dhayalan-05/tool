import os
import sqlite3
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from collections import Counter
from functools import wraps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ------------------------------
# Flask App Setup
# ------------------------------
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

DB_FILE = os.path.join(os.path.dirname(__file__), "activity_data.db")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "url_model.pkl")
VECT_FILE = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
CAT_MODEL_FILE = os.path.join(os.path.dirname(__file__), "cat_model.pkl")
CAT_VECT_FILE = os.path.join(os.path.dirname(__file__), "cat_vectorizer.pkl")

ADMIN_USER = "admin"
ADMIN_PASS = "myStrongPassword123"

# ------------------------------
# Database Setup
# ------------------------------
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

# ------------------------------
# Authentication
# ------------------------------
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

# ============================================================
# ---------- ML Auto-training ----------
# ============================================================
def train_model():
    """Retrains models for flag prediction and category prediction."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT url, flagged, category FROM records")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return

    urls = [r[0] for r in rows]
    flags = np.array([r[1] for r in rows])
    categories = [r[2] for r in rows if r[2]]

    # Flag Prediction Model
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(urls)
    flag_model = LogisticRegression(max_iter=500)
    flag_model.fit(X, flags)
    joblib.dump(flag_model, MODEL_FILE)
    joblib.dump(vectorizer, VECT_FILE)

    # Category Prediction Model retrains automatically from DB too
    if len(set(categories)) > 1:
        cat_vectorizer = TfidfVectorizer(max_features=800)
        X_cat = cat_vectorizer.fit_transform(urls)
        cat_model = LogisticRegression(max_iter=800)
        cat_model.fit(X_cat, categories)
        joblib.dump(cat_model, CAT_MODEL_FILE)
        joblib.dump(cat_vectorizer, CAT_VECT_FILE)

    print(f"✅ ML retrained: {len(urls)} samples")

def predict_flag(url):
    """Predict whether a URL should be flagged."""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECT_FILE):
        return 0
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECT_FILE)
    X = vectorizer.transform([url])
    return int(model.predict(X)[0])

def predict_category(url):
    """Predict category of the URL."""
    # Special case: classify a specific IP as portal
    if "110.171.151.102" in url:
        return "Portal"

    if not os.path.exists(CAT_MODEL_FILE) or not os.path.exists(CAT_VECT_FILE):
        return "Education"

    model = joblib.load(CAT_MODEL_FILE)
    vectorizer = joblib.load(CAT_VECT_FILE)
    X = vectorizer.transform([url])
    return str(model.predict(X)[0])

# ------------------------------
# DB helpers
# ------------------------------
def insert_records(records):
    """Insert new browsing records, predicting category + flag using ML."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    for r in records:
        url = r.get("url", "")
        predicted_flag = predict_flag(url)
        predicted_cat = r.get("category", "") or predict_category(url)
        cur.execute("""
            INSERT INTO records (browser,url,title,timestamp,system_name,lab_name,category,flagged)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            r.get("browser",""),
            url,
            r.get("title",""),
            r.get("timestamp",""),
            r.get("system_name",""),
            r.get("lab_name",""),
            predicted_cat,
            predicted_flag
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
    return [
        {
            "id": r[0],
            "browser": r[1],
            "url": r[2],
            "title": r[3],
            "timestamp": r[4],
            "system_name": r[5],
            "lab_name": r[6],
            "category": r[7],
            "flagged": r[8]
        } for r in rows
    ]

def update_flag(record_id, flagged):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("UPDATE records SET flagged=? WHERE id=?", (flagged, record_id))
    conn.commit()
    conn.close()

# ------------------------------
# API Routes
# ------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "no data"}), 400
    insert_records(data if isinstance(data, list) else [data])
    return jsonify({"status":"ok"}), 200

@app.route("/labs", methods=["GET"])
@requires_auth
def get_labs():
    date = request.args.get("date")
    all_records = fetch_records(date)
    if not all_records:
        return jsonify({}), 200
    lab_map = {}
    for r in all_records:
        lab = r.get("lab_name","Unknown Lab")
        sys = r.get("system_name","Unknown System")
        flag = r.get("flagged",0)
        lab_map.setdefault(lab, {}).setdefault(sys, 0)
        lab_map[lab][sys] = max(lab_map[lab][sys], flag)
    output = {
        lab: [{"system_name": s, "flagged": systems[s]} for s in systems]
        for lab, systems in lab_map.items()
    }
    return jsonify(output),200

@app.route("/data", methods=["GET"])
@requires_auth
def get_data():
    date = request.args.get("date")
    records = fetch_records(date)
    return jsonify(records[-50:]), 200

@app.route("/past_data", methods=["GET"])
@requires_auth
def get_past_data():
    date = request.args.get("date")
    lab = request.args.get("lab")
    system = request.args.get("system")
    records = fetch_records(date)
    if not records:
        return jsonify([]), 200
    if lab:
        records = [r for r in records if r["lab_name"] == lab]
    if system:
        records = [r for r in records if r["system_name"] == system]
    return jsonify(records), 200

@app.route("/flag", methods=["POST"])
@requires_auth
def flag_entry():
    rid = request.get_json(force=True).get("id")
    if rid is None:
        return jsonify({"error":"missing id"}),400
    update_flag(rid, 1)
    train_model()
    return jsonify({"status":"ok"}),200

@app.route("/unflag", methods=["POST"])
@requires_auth
def unflag_entry():
    rid = request.get_json(force=True).get("id")
    if rid is None:
        return jsonify({"error":"missing id"}),400
    update_flag(rid, 0)
    train_model()
    return jsonify({"status":"ok"}),200

@app.route('/')
def dashboard():
    return send_from_directory(os.path.dirname(__file__), 'activity_monitor.html')

# ------------------------------
# Train Category Model from CSV (Auto on startup)
# ------------------------------
def train_from_csv(csv_path):
    """Train or retrain the category model from a CSV (url,title,category)."""
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if 'url' not in df.columns or 'category' not in df.columns:
        print("❌ CSV must contain 'url' and 'category' columns")
        return

    # Ensure balanced dataset
    df = df.groupby('category').apply(lambda x: x.sample(min(len(x), 50))).reset_index(drop=True)

    urls = df['url'].astype(str).tolist()
    categories = df['category'].astype(str).tolist()

    cat_vectorizer = TfidfVectorizer(max_features=800)
    X_cat = cat_vectorizer.fit_transform(urls)
    cat_model = LogisticRegression(max_iter=800)
    cat_model.fit(X_cat, categories)

    joblib.dump(cat_model, CAT_MODEL_FILE)
    joblib.dump(cat_vectorizer, CAT_VECT_FILE)

    print(f"✅ Category model trained successfully on {len(urls)} samples from CSV.")

# ------------------------------
# Run Flask
# ------------------------------
if __name__ == "__main__":
    init_db()
    csv_path = '/home/Dhayalan/url_dataset.csv'
    if os.path.exists(csv_path):
        train_from_csv(csv_path)  # ✅ auto-train category model once at startup
    else:
        print("⚠️ No CSV found, skipping category training")
    train_model()
    app.run(debug=True)
