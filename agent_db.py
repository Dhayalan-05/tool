# Required packages: pip install requests psutil scikit-learn joblib

import os
import sqlite3
import time
import socket
import requests
import shutil
from datetime import datetime, timedelta
import json
import pathlib
import subprocess
import psutil

# --------- CONFIG ----------
SERVER_URL = "https://tool.onrender.com/upload"  # ✅ Your Render URL
AUTH_USER = "admin"
AUTH_PASS = "myStrongPassword123"  # ✅ Must match your Render ADMIN_PASS

SLEEP_INTERVAL = 300  # 5 minutes
LAB_NAME = os.getenv("LAB_NAME", "Lab-1")
SYSTEM_NAME = socket.gethostname()
ALLOWED_DOMAINS = ["110.172.151.102", "your-portal-domain.com"]

VECTOR_FILE = "vectorizer.pkl"
MODEL_FILE = "category_model.pkl"

# ---------- ML SAMPLE DATA ----------
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

# ---------- ML TRAINING ----------
def train_or_load_model():
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier

        if os.path.exists(VECTOR_FILE) and os.path.exists(MODEL_FILE):
            vectorizer = joblib.load(VECTOR_FILE)
            clf = joblib.load(MODEL_FILE)
            print("[Agent ML] Loaded existing model.")
        else:
            print("[Agent ML] Training new model...")
            vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
            X = vectorizer.fit_transform(SAMPLE_TEXTS)
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X, SAMPLE_LABELS)
            joblib.dump(vectorizer, VECTOR_FILE)
            joblib.dump(clf, MODEL_FILE)
            print("[Agent ML] Model trained successfully.")
        return vectorizer, clf
    except Exception as e:
        print(f"[Agent ML] Error in model setup: {e}")
        return None, None

VECTOR, MODEL = train_or_load_model()

def classify_text(text):
    try:
        if not text:
            return "Uncategorized"
        if "110.172.151.102" in text:
            return "Portal"
        if VECTOR is None or MODEL is None:
            return "Uncategorized"
        X = VECTOR.transform([text])
        return MODEL.predict(X)[0]
    except Exception as e:
        print(f"[Agent ML] Prediction error: {e}")
        return "Uncategorized"

# ---------- VPN DETECTION ----------
def detect_and_disable_vpn_adapters():
    """Detect VPN adapters and disable them if active."""
    try:
        ps_cmd = (
            "Get-NetAdapter | Where-Object {($_.Status -eq 'Up') -and "
            "($_.Name -match 'VPN|vpn' -or $_.InterfaceDescription -match 'TAP|OpenVPN|WireGuard|Proton|Nord')} | "
            "Select-Object -ExpandProperty Name"
        )
        result = subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, text=True, timeout=30)
        active_vpns = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        
        if active_vpns:
            for adapter in active_vpns:
                try:
                    disable_cmd = f"Disable-NetAdapter -Name '{adapter}' -Confirm:$false"
                    subprocess.run(["powershell", "-Command", disable_cmd], capture_output=True, timeout=30)
                    print(f"[Agent VPN] Disabled VPN adapter: {adapter}")
                except subprocess.TimeoutExpired:
                    print(f"[Agent VPN] Timeout disabling adapter: {adapter}")
                
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "system": SYSTEM_NAME,
                "lab": LAB_NAME,
                "vpn_disabled": active_vpns
            }
            with open("vpn_log.json", "a", encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")
        else:
            print("[Agent VPN] No active VPN adapters found.")
    except Exception as e:
        print(f"[Agent VPN] Error detecting/disabling VPN: {e}")

# ---------- BROWSER DETECTION ----------
def detect_browsers_all_users():
    users_dir = pathlib.Path("C:/Users")
    browsers_found = {}
    
    if not users_dir.exists():
        print("[Agent] Users directory not found")
        return browsers_found

    for user_folder in users_dir.iterdir():
        if not user_folder.is_dir() or user_folder.name in ["Default", "Public", "All Users"]:
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
                try:
                    for profile in base_path.iterdir():
                        if profile.is_dir() and profile.name not in ["System Profile", "Guest Profile"]:
                            hist = profile / "History"
                            if hist.exists():
                                key = f"{browser_name}_{user_name}_{profile.name}"
                                browsers_found[key] = str(hist)
                except PermissionError:
                    print(f"[Agent] Permission denied accessing {browser_name} for user {user_name}")

        # Firefox
        firefox_path = appdata / "Mozilla/Firefox/Profiles"
        if firefox_path.exists():
            try:
                for profile in firefox_path.iterdir():
                    if profile.is_dir():
                        places = profile / "places.sqlite"
                        if places.exists():
                            key = f"Firefox_{user_name}_{profile.name}"
                            browsers_found[key] = str(places)
            except PermissionError:
                print(f"[Agent] Permission denied accessing Firefox for user {user_name}")
    
    print(f"[Agent] Found {len(browsers_found)} browser profiles")
    return browsers_found

# ---------- HISTORY EXTRACTION ----------
def extract_history(db_path, browser_name):
    temp = f"temp_{hash(db_path) & 0xFFFFFFFF}.db"
    out = []
    
    try:
        # Copy database to avoid locks
        shutil.copy2(db_path, temp)
        
        conn = sqlite3.connect(temp)
        cur = conn.cursor()
        
        # Try Chrome/Edge/Brave schema first
        try:
            cur.execute("SELECT url, title, last_visit_time FROM urls ORDER BY last_visit_time DESC LIMIT 1000")
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            # Try Firefox schema
            try:
                cur.execute("SELECT url, title, last_visit_date FROM moz_places ORDER BY last_visit_date DESC LIMIT 1000")
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                rows = []
        
        conn.close()
        
        for r in rows:
            if not r or len(r) < 3:
                continue
                
            url = r[0] or ""
            title = r[1] or ""
            
            if not url.strip():
                continue
                
            # Convert Chrome timestamp to ISO format if needed
            try:
                if r[2] and r[2] > 10000000000000000:  # Chrome timestamp
                    ts = datetime(1601, 1, 1) + timedelta(microseconds=r[2])
                    timestamp = ts.isoformat()
                else:
                    timestamp = datetime.utcnow().isoformat()
            except:
                timestamp = datetime.utcnow().isoformat()
            
            category = classify_text(f"{title} {url}")
            flagged = 0
            
            # Check if domain is allowed
            if not any(domain in url for domain in ALLOWED_DOMAINS):
                # Simple heuristic for suspicious sites
                suspicious_keywords = ['proxy', 'vpn', 'unblock', 'bypass', 'anonym']
                if any(keyword in url.lower() for keyword in suspicious_keywords):
                    flagged = 1
            
            out.append({
                "browser": browser_name,
                "url": url[:500],  # Limit URL length
                "title": title[:500],  # Limit title length
                "timestamp": timestamp,
                "system_name": SYSTEM_NAME,
                "lab_name": LAB_NAME,
                "category": category,
                "flagged": flagged
            })
            
    except Exception as e:
        print(f"[Agent] {browser_name} extraction error: {e}")
    finally:
        # Cleanup temp file
        try:
            if os.path.exists(temp):
                os.remove(temp)
        except:
            pass
    
    return out

# ---------- BUFFER SYSTEM ----------
BUFFER_FILE = "unsent_buffer.json"

def load_buffer():
    if os.path.exists(BUFFER_FILE):
        try:
            with open(BUFFER_FILE, "r", encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Agent] Error loading buffer: {e}")
            return []
    return []

def save_buffer(data):
    try:
        with open(BUFFER_FILE, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"[Agent] Saved {len(data)} records to buffer")
    except Exception as e:
        print(f"[Agent] Error saving buffer: {e}")

# ---------- UPLOAD SYSTEM ----------
def send_to_server(records, chunk_size=200):
    if not records:
        print("[Agent] No records to send")
        return
        
    all_records = load_buffer() + records
    total = len(all_records)
    sent = 0
    
    print(f"[Agent] Attempting to send {total} records in chunks of {chunk_size}")
    
    while sent < total:
        chunk = all_records[sent:sent + chunk_size]
        try:
            print(f"[Agent] Sending chunk {sent//chunk_size + 1}...")
            
            response = requests.post(
                SERVER_URL,
                json=chunk,
                timeout=30,
                auth=(AUTH_USER, AUTH_PASS),
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                print(f"[Agent] ✅ Successfully sent {len(chunk)} records")
                sent += len(chunk)
            else:
                print(f"[Agent] ❌ Server error {response.status_code}: {response.text}")
                save_buffer(all_records[sent:])
                return
                
        except requests.exceptions.ConnectionError:
            print(f"[Agent] ❌ Connection failed - server unreachable")
            save_buffer(all_records[sent:])
            return
        except requests.exceptions.Timeout:
            print(f"[Agent] ⚠ Request timeout")
            save_buffer(all_records[sent:])
            return
        except Exception as e:
            print(f"[Agent] ❌ Upload error: {e}")
            save_buffer(all_records[sent:])
            return
    
    # Clear buffer if all sent successfully
    if os.path.exists(BUFFER_FILE):
        os.remove(BUFFER_FILE)
        print("[Agent] ✅ Buffer cleared")

# ---------- MAIN LOOP ----------
def main():
    print(f"=== Agent Started ===\nSystem: {SYSTEM_NAME}\nLab: {LAB_NAME}")
    print(f"Target: {SERVER_URL}")
    
    iteration = 0
    while True:
        try:
            print(f"\n--- Iteration {iteration} ---")
            
            # Run VPN detection every 10 iterations
            if iteration % 10 == 0:
                detect_and_disable_vpn_adapters()
            
            # Collect browser data
            browsers = detect_browsers_all_users()
            aggregated = []
            
            for bname, path in browsers.items():
                records = extract_history(path, bname)
                aggregated.extend(records)
                print(f"[Agent] Extracted {len(records)} records from {bname}")
            
            print(f"[Agent] Total collected: {len(aggregated)} records")
            
            # Send to server
            if aggregated:
                send_to_server(aggregated)
            else:
                print("[Agent] No new records to send")
            
            # Check buffer size
            buffer_size = len(load_buffer())
            if buffer_size > 0:
                print(f"[Agent] ⚠ Unsent records in buffer: {buffer_size}")
            
            iteration += 1
            print(f"[Agent] Waiting {SLEEP_INTERVAL} seconds...")
            time.sleep(SLEEP_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n[Agent] Stopped by user")
            break
        except Exception as e:
            print(f"[Agent] Error in main loop: {e}")
            time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
