# agent_db.py
# Required packages: pip install requests psutil scikit-learn joblib

import os
import sqlite3
import time
import socket
import requests
import shutil
from datetime import datetime
import json

SERVER_URL = "http://127.0.0.1:8000/upload"  # replace with admin server IP if needed
SLEEP_INTERVAL = 60
LAB_NAME = os.getenv("LAB_NAME", "Lab-1")
SYSTEM_NAME = socket.gethostname()

ALLOWED_DOMAINS = ["collegeportal.edu"]  # domains not flagged

VECTOR_FILE = "vectorizer.pkl"
MODEL_FILE = "category_model.pkl"

SAMPLE_TEXTS = [
    "facebook.com","youtube.com/watch?v=abc","instagram.com/user",
    "wikipedia.org/wiki/Python","stackoverflow.com/questions","github.com/repo",
    "netflix.com/title","coursera.org/course","khanacademy.org",
    "google.com/search?q=python","docs.python.org","linkedin.com","gmail.com","zoom.us"
]
SAMPLE_LABELS = [
    "Social","Entertainment","Social",
    "Educational","Technical","Technical",
    "Entertainment","Educational","Educational",
    "Search","Technical","Work","Communication","Communication"
]

# ---------- ML helpers ----------
def train_or_load_model():
    import joblib
    try:
        vectorizer = joblib.load(VECTOR_FILE)
        clf = joblib.load(MODEL_FILE)
        return vectorizer, clf
    except:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        vectorizer = TfidfVectorizer(ngram_range=(1,2))
        X = vectorizer.fit_transform(SAMPLE_TEXTS)
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, SAMPLE_LABELS)
        joblib.dump(vectorizer, VECTOR_FILE)
        joblib.dump(clf, MODEL_FILE)
        return vectorizer, clf

VECTOR, MODEL = train_or_load_model()

def classify_text(text):
    try:
        X = VECTOR.transform([text])
        return MODEL.predict(X)[0]
    except:
        return "Other"

# ---------- Browser detection ----------
def detect_browsers():
    browsers_found = {}
    local = os.getenv("LOCALAPPDATA", "")
    appdata = os.getenv("APPDATA", "")

    if local:
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
            places = os.path.join(firefox_root, profile, "places.sqlite")
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
        try:
            cur.execute("SELECT url, title, last_visit_time FROM urls ORDER BY last_visit_time DESC LIMIT 100")
            rows = cur.fetchall()
        except:
            try:
                cur.execute("SELECT url, title, last_visit_date FROM moz_places ORDER BY last_visit_date DESC LIMIT 100")
                rows = cur.fetchall()
            except:
                rows=[]
        conn.close()
        os.remove(temp)
        for r in rows:
            url = r[0] if r and len(r) > 0 else ""
            title = r[1] if r and len(r) > 1 else ""
            ts = datetime.utcnow().isoformat()
            category = classify_text(title + " " + url)
            flagged = 0
            if not any(d in url for d in ALLOWED_DOMAINS):
                flagged = 1
            out.append({
                "browser": browser_name,
                "url": url,
                "title": title,
                "timestamp": ts,
                "system_name": SYSTEM_NAME,
                "lab_name": LAB_NAME,
                "category": category,
                "flagged": flagged
            })
    except Exception as e:
        print(f"[Agent] {browser_name} extraction error:", e)
    return out

BUFFER_FILE = "unsent_buffer.json"

def load_buffer():
    if os.path.exists(BUFFER_FILE):
        try:
            with open(BUFFER_FILE,"r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_buffer(data):
    with open(BUFFER_FILE,"w") as f:
        json.dump(data,f)

def send_to_server(records):
    if not records:
        return
    all_records = load_buffer() + records
    try:
        res = requests.post(SERVER_URL,json=all_records,timeout=10)
        if res.status_code == 200:
            print(f"[Agent] Sent {len(all_records)} records to server.")
            if os.path.exists(BUFFER_FILE):
                os.remove(BUFFER_FILE)
        else:
            save_buffer(all_records)
    except:
        save_buffer(all_records)

def main():
    print("=== Agent started ===")
    while True:
        aggregated=[]
        browsers = detect_browsers()
        for bname, path in browsers.items():
            aggregated.extend(extract_history(path,bname))
        send_to_server(aggregated)
        time.sleep(SLEEP_INTERVAL)

if __name__=="__main__":
    main()
