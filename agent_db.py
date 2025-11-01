# Required packages: pip install requests psutil scikit-learn joblib

import os
import sqlite3
import time
import socket
import requests
import shutil
from datetime import datetime
import json
import pathlib
import subprocess  # ✅ for PowerShell VPN control

# --------- CONFIG ----------
SERVER_URL = "https://tool.onrender.com/upload"  # ✅ Updated to your Render backend
AUTH_USER = "admin"
AUTH_PASS = "myStrongPassword123"

SLEEP_INTERVAL = 60
LAB_NAME = os.getenv("LAB_NAME", "Lab-1")
SYSTEM_NAME = socket.gethostname()
ALLOWED_DOMAINS = ["110.172.151.102"]

VECTOR_FILE = "vectorizer.pkl"
MODEL_FILE = "category_model.pkl"

# ---------- ✅ ML SAMPLE DATA ----------
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

# ---------- ✅ ML TRAINING ----------
def train_or_load_model():
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
    try:
        if "110.172.151.102" in text:
            return "Portal"
        if VECTOR is None or MODEL is None:
            return "Uncategorized"
        X = VECTOR.transform([text])
        return MODEL.predict(X)[0]
    except Exception as e:
        print(f"[Agent ML] Prediction error: {e}")
        return "Uncategorized"

# ---------- ✅ VPN DETECTION ----------
def detect_and_disable_vpn_adapters():
    """Detect VPN adapters and disable them if active."""
    try:
        ps_cmd = (
            "Get-NetAdapter | Where-Object {($_.Status -eq 'Up') -and "
            "($_.Name -match 'VPN' -or $_.InterfaceDescription -match 'TAP|OpenVPN|WireGuard|Proton|Nord')} | "
            "Select-Object -ExpandProperty Name"
        )
        result = subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, text=True)
        active_vpns = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if active_vpns:
            for adapter in active_vpns:
                disable_cmd = f"Disable-NetAdapter -Name '{adapter}' -Confirm:$false"
                subprocess.run(["powershell", "-Command", disable_cmd], capture_output=True)
                print(f"[Agent VPN] Disabled VPN adapter: {adapter}")
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "system": SYSTEM_NAME,
                "lab": LAB_NAME,
                "vpn_disabled": active_vpns
            }
            with open("vpn_log.json", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        else:
            print("[Agent VPN] No active VPN adapters found.")
    except Exception as e:
        print(f"[Agent VPN] Error detecting/disabling VPN: {e}")

# ---------- ✅ BROWSER DETECTION ----------
def detect_browsers_all_users():
    users_dir = pathlib.Path("C:/Users")
    browsers_found = {}
    for user_folder in users_dir.iterdir():
        if not user_folder.is_dir():
            continue
        user_name = user_folder.name
        local = user_folder / "AppData/Local"
        appdata = user_folder / "AppData/Roaming"

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

        firefox_path = appdata / "Mozilla/Firefox/Profiles"
        if firefox_path.exists():
            for profile in firefox_path.iterdir():
                places = profile / "places.sqlite"
                if places.exists():
                    browsers_found[f"Firefox_{user_name}_{profile.name}"] = str(places)
    return browsers_found

# ---------- ✅ HISTORY EXTRACTION ----------
def extract_history(db_path, browser_name):
    temp = f"temp_{browser_name.replace(' ', '_')}.db"
    out = []
    try:
        shutil.copy2(db_path, temp)
        conn = sqlite3.connect(temp)
        cur = conn.cursor()
        try:
            cur.execute("SELECT url, title, last_visit_time FROM urls ORDER BY last_visit_time DESC")
            rows = cur.fetchall()
        except:
            try:
                cur.execute("SELECT url, title, last_visit_date FROM moz_places ORDER BY last_visit_date DESC")
                rows = cur.fetchall()
            except:
                rows = []
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

# ---------- ✅ BUFFER SYSTEM ----------
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

# ---------- ✅ CHUNKED UPLOAD ----------
def send_to_server(records, chunk_size=500):
    if not records:
        return
    all_records = load_buffer() + records
    total = len(all_records)
    sent = 0
    while sent < total:
        chunk = all_records[sent:sent + chunk_size]
        try:
            res = requests.post(
                SERVER_URL,
                json=chunk,
                timeout=60,
                auth=(AUTH_USER, AUTH_PASS)
            )
            if res.status_code == 200:
                print(f"[Agent] ✅ Sent {len(chunk)} records (chunk {sent//chunk_size+1}).")
            else:
                print(f"[Agent] ⚠ Server returned {res.status_code}, buffering chunk.")
                save_buffer(all_records[sent:])
                return
        except Exception as e:
            print(f"[Agent] ⚠ Send failed: {e}, buffering remaining data.")
            save_buffer(all_records[sent:])
            return
        sent += len(chunk)
    if os.path.exists(BUFFER_FILE):
        os.remove(BUFFER_FILE)

# ---------- ✅ MAIN LOOP ----------
def main():
    print(f"=== Agent Started ===\nSending data to: {SERVER_URL}")
    while True:
        detect_and_disable_vpn_adapters()
        aggregated = []
        browsers = detect_browsers_all_users()
        for bname, path in browsers.items():
            aggregated.extend(extract_history(path, bname))
        print(f"[Agent] Collected {len(aggregated)} records.")
        send_to_server(aggregated)
        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
