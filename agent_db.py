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

# ---------- TRAINING DATA ----------
SAMPLE_TEXTS = [
    "facebook.com", "youtube.com/watch?v=abc123", "instagram.com/user/profile",
    "stackoverflow.com/questions/12345", "github.com/openai/gpt", "netflix.com/title/6789",
    "google.com/search?q=machine+learning", "docs.python.org/3/tutorial/", "linkedin.com/in/developer",
    "gmail.com/mail/u/0", "zoom.us/j/123456789", "twitter.com/home", "reddit.com/r/technology/",
    "spotify.com/playlist/abcd", "coursera.org/learn/python", "medium.com/@techstories",
    "hulu.com/watch/series", "dropbox.com/s/files", "slack.com/workspace", "bbc.com/news/world",
    "pinterest.com/ideas/", "amazon.in/product/B09XYZ", "office.com/login", "telegram.me/channel",
    "twitch.tv/livegame", "openai.com/research", "disneyplus.com/movies/", "flipkart.com/viewcart",
    "nytimes.com/section/technology", "quora.com/topic/AI", "udemy.com/course/python",
    "hotstar.com/in", "whatsapp.com", "teams.microsoft.com", "news.google.com", "medium.com",
    "developer.mozilla.org", "microsoft.com", "trello.com", "notion.so", "zoom.us", "snapchat.com"
]

SAMPLE_LABELS = [
    "Social", "Entertainment", "Social",
    "Technical", "Technical", "Entertainment",
    "Search", "Technical", "Work",
    "Communication", "Communication", "Social", "Social",
    "Entertainment", "Education", "Blog",
    "Entertainment", "Work", "Work", "News",
    "Social", "E-commerce", "Work", "Communication",
    "Entertainment", "Technical", "Entertainment",
    "E-commerce", "News", "Knowledge", "Education",
    "Entertainment", "Social", "Work", "News", "Blog",
    "Technical", "Work", "Work", "Work", "Communication", "Social"
]

# ---------- ML MODEL ----------
def train_or_load_model():
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    try:
        vectorizer = joblib.load(VECTOR_FILE)
        clf = joblib.load(MODEL_FILE)
        return vectorizer, clf
    except:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', lowercase=True)
        X = vectorizer.fit_transform(SAMPLE_TEXTS)
        clf = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
        clf.fit(X, SAMPLE_LABELS)
        joblib.dump(vectorizer, VECTOR_FILE)
        joblib.dump(clf, MODEL_FILE)
        print("[Agent] ML model trained and saved.")
        return vectorizer, clf

VECTOR, MODEL = train_or_load_model()

def classify_text(text):
    try:
        X = VECTOR.transform([text])
        prediction = MODEL.predict(X)[0]
        return prediction
    except Exception as e:
        print("[Agent] Classification error:", e)
        return "Other"

# ---------- BROWSER DETECTION ----------
def detect_browsers():
    browsers_found = {}
    local = os.getenv("LOCALAPPDATA", "")
    appdata = os.getenv("APPDATA", "")

    if local:
        for folder in os.listdir(local):
            path = os.path.join(local, folder)
            if not os.path.isdir(path):
                continue

            chrome_path = os.path.join(path, "Google", "Chrome", "User Data")
            if os.path.exists(chrome_path):
                for profile in os.listdir(chrome_path):
                    hist = os.path.join(chrome_path, profile, "History")
                    if os.path.exists(hist):
                        browsers_found[f"Chrome_{profile}"] = hist

            edge_path = os.path.join(path, "Microsoft", "Edge", "User Data")
            if os.path.exists(edge_path):
                for profile in os.listdir(edge_path):
                    hist = os.path.join(edge_path, profile, "History")
                    if os.path.exists(hist):
                        browsers_found[f"Edge_{profile}"] = hist

            brave_path = os.path.join(path, "BraveSoftware", "Brave-Browser", "User Data")
            if os.path.exists(brave_path):
                for profile in os.listdir(brave_path):
                    hist = os.path.join(brave_path, profile, "History")
                    if os.path.exists(hist):
                        browsers_found[f"Brave_{profile}"] = hist

    firefox_root = os.path.join(appdata, "Mozilla", "Firefox", "Profiles")
    if os.path.exists(firefox_root):
        for profile in os.listdir(firefox_root):
            places = os.path.join(firefox_root, profile, "places.sqlite")
            if os.path.exists(places):
                browsers_found[f"Firefox_{profile}"] = places

    return browsers_found

# ---------- HISTORY EXTRACTION ----------
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
            cur.execute("SELECT url, title, last_visit_date FROM moz_places ORDER BY last_visit_date DESC LIMIT 100")
            rows = cur.fetchall()

        conn.close()
        os.remove(temp)

        for r in rows:
            url = r[0] if r else ""
            title = r[1] if len(r) > 1 else ""
            ts = datetime.utcnow().isoformat()
            category = classify_text(title + " " + url)
            flagged = 0 if any(d in url for d in ALLOWED_DOMAINS) else 1

            print(f"[Agent ML] {url} => {category}")


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

# ---------- DATA BUFFER ----------
BUFFER_FILE = "unsent_buffer.json"

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

# ---------- SERVER UPLOAD ----------
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
            print("[Agent] ⚠ Server rejected data:", res.status_code)
            save_buffer(all_records)
    except Exception as e:
        print(f"[Agent] ⚠ Could not send data to server: {e}")
        save_buffer(all_records)

# ---------- MAIN ----------
def main():
    print(f"=== Agent started ===\nSending data to: {SERVER_URL}")
    while True:
        aggregated = []
        browsers = detect_browsers()
        for bname, path in browsers.items():
            aggregated.extend(extract_history(path, bname))
        send_to_server(aggregated)
        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
