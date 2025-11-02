import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from functools import wraps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pymongo import MongoClient
from bson.objectid import ObjectId

# ------------------------------
# Flask App Setup
# ------------------------------
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ------------------------------
# MongoDB connection
# ------------------------------
# URL-encode special characters in your MongoDB password
# Password: (dhayalan@05) -> %28dhayalan%4005%29
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://dhayalan:%28Dhayalan%2405%29@browser.uwvjryb.mongodb.net/?retryWrites=true&w=majority"
)

client = MongoClient(MONGO_URI)
db = client["activity_db"]
records_col = db["records"]

# ------------------------------
# ML Model Files
# ------------------------------
MODEL_FILE = os.path.join(os.path.dirname(__file__), "url_model.pkl")
VECT_FILE = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
CAT_MODEL_FILE = os.path.join(os.path.dirname(__file__), "cat_model.pkl")
CAT_VECT_FILE = os.path.join(os.path.dirname(__file__), "cat_vectorizer.pkl")

# ------------------------------
# Admin Authentication
# ------------------------------
ADMIN_USER = "admin"
ADMIN_PASS = "myStrongPassword123"

def check_auth(username, password):
    return username == ADMIN_USER and password == ADMIN_PASS

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response('Login required', 401,
                            {'WWW-Authenticate': 'Basic realm="Login Required"'})
        return f(*args, **kwargs)
    return decorated

# ------------------------------
# ML Functions
# ------------------------------
def train_model():
    """Retrain models for flag prediction and category prediction."""
    rows = list(records_col.find({}, {"url": 1, "flagged": 1, "category": 1, "_id": 0}))
    if not rows:
        return

    urls = [r["url"] for r in rows if "url" in r]
    flags = np.array([r.get("flagged", 0) for r in rows])
    categories = [r.get("category") for r in rows if r.get("category")]

    # Flag prediction
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(urls)
    flag_model = LogisticRegression(max_iter=500)
    flag_model.fit(X, flags)
    joblib.dump(flag_model, MODEL_FILE)
    joblib.dump(vectorizer, VECT_FILE)

    # Category prediction
    if len(set(categories)) > 1:
        cat_vectorizer = TfidfVectorizer(max_features=800)
        X_cat = cat_vectorizer.fit_transform(urls)
        cat_model = LogisticRegression(max_iter=800)
        cat_model.fit(X_cat, categories)
        joblib.dump(cat_model, CAT_MODEL_FILE)
        joblib.dump(cat_vectorizer, CAT_VECT_FILE)

    print(f"✅ ML retrained: {len(urls)} samples")

def predict_flag(url):
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECT_FILE):
        return 0
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECT_FILE)
    X = vectorizer.transform([url])
    return int(model.predict(X)[0])

def predict_category(url):
    if not os.path.exists(CAT_MODEL_FILE) or not os.path.exists(CAT_VECT_FILE):
        return "Uncategorized"
    model = joblib.load(CAT_MODEL_FILE)
    vectorizer = joblib.load(CAT_VECT_FILE)
    X = vectorizer.transform([url])
    return str(model.predict(X)[0])

# ------------------------------
# MongoDB Helpers
# ------------------------------
def insert_records(records):
    docs = []
    for r in records:
        url = r.get("url", "")
        predicted_flag = predict_flag(url)
        predicted_cat = r.get("category") or predict_category(url)
        docs.append({
            "browser": r.get("browser", ""),
            "url": url,
            "title": r.get("title", ""),
            "timestamp": r.get("timestamp", ""),
            "system_name": r.get("system_name", ""),
            "lab_name": r.get("lab_name", ""),
            "category": predicted_cat,
            "flagged": predicted_flag
        })
    if docs:
        records_col.insert_many(docs)

def fetch_records(date=None):
    query = {}
    if date:
        query["timestamp"] = {"$regex": f"^{date}"}
    rows = list(records_col.find(query))
    return [
        {
            "id": str(r["_id"]),
            "browser": r.get("browser", ""),
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "timestamp": r.get("timestamp", ""),
            "system_name": r.get("system_name", ""),
            "lab_name": r.get("lab_name", ""),
            "category": r.get("category", ""),
            "flagged": r.get("flagged", 0)
        } for r in rows
    ]

def update_flag(record_id, flagged):
    records_col.update_one({"_id": ObjectId(record_id)}, {"$set": {"flagged": flagged}})

# ------------------------------
# Flask Routes
# ------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "no data"}), 400
    insert_records(data if isinstance(data, list) else [data])
    return jsonify({"status": "ok"}), 200

@app.route("/labs", methods=["GET"])
@requires_auth
def get_labs():
    date = request.args.get("date")
    all_records = fetch_records(date)
    if not all_records:
        return jsonify({})
    lab_map = {}
    for r in all_records:
        lab = r.get("lab_name", "Unknown Lab")
        sys = r.get("system_name", "Unknown System")
        flag = r.get("flagged", 0)
        lab_map.setdefault(lab, {}).setdefault(sys, 0)
        lab_map[lab][sys] = max(lab_map[lab][sys], flag)
    return jsonify({
        lab: [{"system_name": s, "flagged": systems[s]} for s in systems]
        for lab, systems in lab_map.items()
    })

@app.route("/data", methods=["GET"])
@requires_auth
def get_data():
    date = request.args.get("date")
    records = fetch_records(date)
    return jsonify(records[-50:])

@app.route("/flag", methods=["POST"])
@requires_auth
def flag_entry():
    rid = request.get_json(force=True).get("id")
    if not rid:
        return jsonify({"error": "missing id"}), 400
    update_flag(rid, 1)
    train_model()
    return jsonify({"status": "ok"})

@app.route("/unflag", methods=["POST"])
@requires_auth
def unflag_entry():
    rid = request.get_json(force=True).get("id")
    if not rid:
        return jsonify({"error": "missing id"}), 400
    update_flag(rid, 0)
    train_model()
    return jsonify({"status": "ok"})

@app.route('/')
def dashboard():
    return send_from_directory(os.path.dirname(__file__), 'activity_monitor.html')

# ------------------------------
# Run Flask
# ------------------------------
if __name__ == "__main__":
    train_model()  # Initial training
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
