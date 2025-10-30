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
import getpass
import pathlib

# --------- CONFIG ----------
SERVER_URL = "https://Dhayalan.pythonanywhere.com/upload"
AUTH_USER = "admin"
AUTH_PASS = "myStrongPassword123"

SLEEP_INTERVAL = 60
LAB_NAME = os.getenv("LAB_NAME", "Lab-1")
SYSTEM_NAME = socket.gethostname()
ALLOWED_DOMAINS = ["110.172.151.102"]

VECTOR_FILE = "vectorizer.pkl"
MODEL_FILE = "category_model.pkl"

# ✅ FIXED ML TRAINING DATA
SAMPLE_TEXTS = [
    "facebook.com","youtube.com/watch?v=abc123","instagram.com/user/profile",
    "stackoverflow.com/questions/12345","github.com/openai/gpt","netflix.com/title/6789",
    "google.com/search?q=machine+learning","docs.python.org/3/tutorial/","linkedin.com/in/developer",
    "gmail.com/mail/u/0","zoom.us/j/123456789","twitter.com/home","reddit.com/r/technology/",
    "spotify.com/playlist/abcd","coursera.org/learn/python","medium.com/@techstories",
    "hulu.com/watch/series","dropbox.com/s/files","slack.com/workspace","bbc.com/news/world",
    "pinterest.com/ideas/","amazon.in/product/B09XYZ","office.com/login","telegram.me/channel",
    "twitch.tv/livegame","openai.com/research","disneyplus.com/movies/","flipkart.com/viewcart",
    "nytimes.com/section/technology","quora.com/topic/AI","wikipedia.org/wiki/Python"
]

SAMPLE_LABELS = [
    "Social","Entertainment","Social","Technical","Technical","Entertainment",
    "Search","Technical","Work","Communication","Communication","Social","Social",
    "Entertainment","Education","Blog","Entertainment","Work","Work","News",
    "Social","E-commerce","Work","Communication","Entertainment","Technical",
    "Entertainment","E-commerce","News","Knowledge","Education"
]

# ---------- ✅ FIXED ML ----------
def train_or_load_model():
    """Train or load model safely without mismatch"""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier

    try:
        if os.path.exists(VECTOR_FILE) and os.path.exists(MODEL_FILE):
            vectorizer = joblib.load(VECTOR_FILE)
            clf = joblib.load(MODEL_FILE)
            print("[Agent ML] Loaded existing model.")
        else:
            print("[Agent ML] Training new model...")
            vectorizer = TfidfVectorizer(ngram_range=(1,2))
            X = vectorizer.fit_transform(SAMPLE_TEXTS)
            clf = RandomForestClassifier(n_estimators=80, random_state=42)
            clf.fit(X, SAMPLE_LABELS)
            joblib.dump(vectorizer, VECTOR_FILE)
            joblib.dump(clf, MODEL_FILE)
            print("[Agent ML] Model trained successfully.")
        return vectorizer, clf
    except Exception as e:
        print(f"[Agent ML] Error training model: {e}")
        return None, None

VECTOR, MODEL = train_or_load_model()

def classify_text(text):
    """Predict URL category"""
    try:
        if VECTOR is None or MODEL is None:
            return "Uncategorized"
        X = VECTOR.transform([text])
        return MODEL.predict(X)[0]
    except Exception as e:
        print(f"[Agent ML] Prediction error: {e}")
        return "Uncategorized"

# ---------- ✅ MULTI-USER BROWSER DETECTION ----------
def detect_browsers_all_users():
    """Detect browsers for all user accounts in Windows"""
    users_dir = pathlib.Path("C:/Users")
    browsers_found = {}

    for user_folder in users_dir.iterdir():
        if not user_folder.is_dir():
            continue
        user_name = user_folder.name
        local = user_folder / "AppData/Local"
        appdata = user_folder / "AppData/Roaming"

        # Chrome, Edge, Brave
        for browser_name, subpath in {
            "Chrome": "Google/Chrome/User Data",
            "Edge": "Microsoft/Edge/User Data",
            "Brave": "BraveSoftware/Brave-Browser/User Data"
        }.items():
            base_path = local / subpath
            if base_path.exists():
                for profile in base_path.iterdir():
                    hist = profile / "History"
                    if hist.exists():
                        browsers_found[f"{browser_name}_{user_name}_{profile.name}"] = str(hist)

        # Firefox
        firefox_path = appdata / "Mozilla/Firefox/Profiles"
        if firefox_path.exists():
            for profile in firefox_path.iterdir():
                places = profile / "places.sqlite"
                if places.exists():
                    browsers_found[f"Firefox_{user_name}_{profile.name}"] = str(places)

    return browsers_found

# ---------- Browser Extraction ----------
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

        # Extract username from browser name tag
        user_tag = "Unknown"
        parts = browser_name.split("_")
        if len(parts) >= 2:
            user_tag = parts[1]

        for r in rows:
            url = r[0] if r and len(r)>0 else ""
            title = r[1] if r and len(r)>1 else ""
            ts = datetime.utcnow().isoformat()
            category = classify_text(title + " " + url)
            flagged = 0
            if not any(d in url for d in ALLOWED_DOMAINS):
                flagged = 1
            out.append({
                "browser": browser_name,
                "user_name": user_tag,
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
        res = requests.post(SERVER_URL, json=all_records, timeout=10, auth=(AUTH_USER, AUTH_PASS))
        if res.status_code == 200:
            print(f"[Agent] Sent {len(all_records)} records to server.")
            if os.path.exists(BUFFER_FILE):
                os.remove(BUFFER_FILE)
        else:
            save_buffer(all_records)
    except Exception as e:
        print(f"[Agent] ⚠ Could not send data to server: {e}")
        save_buffer(all_records)

def main():
    print(f"=== Agent started ===\nSending data to: {SERVER_URL}")
    while True:
        aggregated=[]
        browsers = detect_browsers_all_users()
        for bname, path in browsers.items():
            aggregated.extend(extract_history(path,bname))
        send_to_server(aggregated)
        time.sleep(SLEEP_INTERVAL)

if __name__=="__main__":
    main()
