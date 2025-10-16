import os
import socket
import json
import time
import requests
from datetime import datetime
import sqlite3

SERVER_URL = "http://10.1.7.84:8000/upload"
 # your Flask server
SYSTEM_NAME = socket.gethostname()  # system name like LAB-PC1

DB_PATH = "browser_history.db"

def fetch_browser_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, browser, url, title, timestamp FROM activities")
    rows = cursor.fetchall()
    conn.close()

    data = []
    for row in rows:
        data.append({
            "id": row[0],
            "browser": row[1],
            "url": row[2],
            "title": row[3],
            "timestamp": row[4],
            "system_name": SYSTEM_NAME
        })
    return data

def send_data_to_server():
    try:
        data = fetch_browser_data()
        if data:
            res = requests.post(SERVER_URL, json=data)
            print(f"[Agent] Sent {len(data)} records - Server responded {res.status_code}")
    except Exception as e:
        print("[Agent] Error:", e)

if __name__ == "__main__":
    while True:
        send_data_to_server()
        time.sleep(30)  # send every 30 seconds
