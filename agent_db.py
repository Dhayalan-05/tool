# agent_db.py
# Required packages:
# pip install requests psutil scikit-learn joblib

import os
import sqlite3
import time
import psutil
import socket
import requests
import shutil
from datetime import datetime

# CONFIG
SERVER_URL = "http://127.0.0.1:8000/upload"  # change to admin IP when deploying
SLEEP_INTERVAL = 60  # seconds between extraction cycles
LAB_NAME = os.getenv("LAB_NAME", "Lab-1")  # set LAB_NAME env var on different labs if needed
SYSTEM_NAME = socket.gethostname()

# ML model filenames
VECT_FILE = "vectorizer.pkl"
MODEL_FILE = "category_model.pkl"

# Minimal sample training data (used only if no trained model)
SAMPLE_TEXTS = [
    "facebook.com", "youtube.com/watch?v=abc", "instagram.com/user",
    "wikipedia.org/wiki/Python", "stackoverflow.com/questions", "github.com/repo",
    "netflix.com/title", "coursera.org/course", "khanacademy.org",
    "google.com/search?q=python", "docs.python.org", "linkedin.com", "gmail.com", "zoom.us"
]
SAMPLE_LABELS = [
    "Social", "Entertainment", "Social",
    "Educational", "Technical", "Technical",
    "Entertainment", "Educational", "Educational",
    "Search", "Technical", "Work", "Communication", "Communication"
]

CATEGORIES = list(set(SAMPLE_LABELS))

# ---------- ML helpers ----------
def train_or_load_model():
    try:
        import joblib
        vectorizer = joblib.load(VECT_FILE)
        clf = joblib.load(MODEL_FILE)
        return vectorizer, clf
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        vectorizer = TfidfVectorizer(ngram_range=(1,2))
        X = vectorizer.fit_transform(SAMPLE_TEXTS)
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, SAMPLE_LABELS)
        import joblib
        joblib.dump(vectorizer, VECT_FILE)
        joblib.dump(clf, MODEL_FILE)
        print("[ML] Trained initial model and saved to disk.")
        return vectorizer, clf

VECTOR, MODEL = train_or_load_model()

def classify_text(text):
    try:
        X = VECTOR.transform([text])
        return MODEL.predict(X)[0]
    except Exception:
        return "Other"

# ---------- Browser detection & extraction ----------
def detect_browsers():
    browsers_found = {}
    local = os.getenv("LOCALAPPDATA", "")
    appdata = os.getenv("APPDATA", "")

    # Dynamic search in LocalAppData for Chrome, Edge, Brave
    if local and os.path.exists(local):
        for folder in os.listdir(local):
            path = os.path.join(local, folder)
            if os.path.isdir(path):
                # Chrome
                chrome_path = os.path.join(path, "Google\\Chrome\\User Data")
                if os.path.exists(chrome_path):
                    for profile in os.listdir(chrome_path):
                        hist = os.path.join(chrome_path, profile, "History")
                        if os.path.exists(hist):
                            browsers_found[f"Chrome_{profile}"] = hist
                # Edge
                edge_path = os.path.join(path, "Microsoft\\Edge\\User Data")
                if os.path.exists(edge_path):
                    for profile in os.listdir(edge_path):
                        hist = os.path.join(edge_path, profile, "History")
                        if os.path.exists(hist):
                            browsers_found[f"Edge_{profile}"] = hist
                # Brave
                brave_path = os.path.join(path, "BraveSoftware\\Brave-Browser\\User Data")
                if os.path.exists(brave_path):
                    for profile in os.listdir(brave_path):
                        hist = os.path.join(brave_path, profile, "History")
                        if os.path.exists(hist):
                            browsers_found[f"Brave_{profile}"] = hist

    # Firefox
    firefox_root = os.path.join(appdata, "Mozilla\\Firefox\\Profiles")
    if firefox_root and os.path.exists(firefox_root):
        for profile in os.listdir(firefox_root):
            profile_dir = os.path.join(firefox_root, profile)
            places = os.path.join(profile_dir, "places.sqlite")
            if os.path.exists(places):
                browsers_found[f"Firefox_{profile}"] = places

    return browsers_found

def extract_history(db_path, browser_name):
    temp = f"temp_{browser_name.replace(' ', '_')}.db"
    out = []
    try:
        shutil.copy2(db_path, temp)
        conn = sqlite3.connect(temp)
        cur = conn.cursor()
        # Try Chrome/Edge/Brave style
        try:
            cur.execute("SELECT url, title, last_visit_time FROM urls ORDER BY last_visit_time DESC LIMIT 100")
            rows = cur.fetchall()
        except Exception:
            try:
                cur.execute("SELECT url, title, last_visit_date FROM moz_places ORDER BY last_visit_date DESC LIMIT 100")
                rows = cur.fetchall()
            except Exception:
                rows = []
        conn.close()
        os.remove(temp)
        for r in rows:
            url = r[0] if r and len(r) > 0 else ""
            title = r[1] if r and len(r) > 1 else ""
            ts = datetime.utcnow().isoformat()
            category = classify_text(title + " " + url)
            out.append({
                "browser": browser_name,
                "url": url,
                "title": title,
                "timestamp": ts,
                "system_name": SYSTEM_NAME,
                "lab_name": LAB_NAME,
                "category": category
            })
    except Exception as e:
        print(f"[extract_history] {browser_name} error: {e}")
    return out

# ---------- Buffer for offline storage ----------
BUFFER_FILE = "unsent_buffer.json"
import json

def load_buffer():
    if os.path.exists(BUFFER_FILE):
        try:
            with open(BUFFER_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_buffer(data):
    with open(BUFFER_FILE, "w") as f:
        json.dump(data, f)

# ---------- Send to server ----------
def send_to_server(records):
    if not records:
        return
    all_records = load_buffer() + records
    try:
        headers = {"Content-Type":"application/json"}
        res = requests.post(SERVER_URL, json=all_records, headers=headers, timeout=10)
        if res.status_code == 200:
            print(f"[Agent] Sent {len(all_records)} records to server.")
            if os.path.exists(BUFFER_FILE):
                os.remove(BUFFER_FILE)
        else:
            print(f"[Agent] Server responded {res.status_code}: {res.text}")
            save_buffer(all_records)
    except Exception as e:
        print(f"[Agent] Send failed: {e}")
        save_buffer(all_records)

# ---------- Main loop ----------
def main():
    print("=== Agent started ===")
    while True:
        try:
            aggregated = []
            browsers = detect_browsers()
            for bname, path in browsers.items():
                extracted = extract_history(path, bname)
                if extracted:
                    aggregated.extend(extracted)
            send_to_server(aggregated)
        except Exception as e:
            print("[Agent] Error in main loop:", e)
        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
