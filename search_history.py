import os
import sqlite3
import csv
import shutil
from pathlib import Path
import datetime

def get_chrome_history():
    history_path = os.path.expanduser(
        "~\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\History"
    )
    
    # Copy the file because Chrome locks it while running
    temp_history = "History_temp"
    try:
        shutil.copy2(history_path, temp_history)
    except Exception as e:
        print("Error copying Chrome history:", e)
        return []

    urls = []
    try:
        conn = sqlite3.connect(temp_history)
        cursor = conn.cursor()
        cursor.execute("SELECT url, title, last_visit_time FROM urls")
        for url, title, last_visit_time in cursor.fetchall():
            # Convert Chrome timestamp to human-readable
            timestamp = datetime.datetime(1601, 1, 1) + datetime.timedelta(microseconds=last_visit_time)
            urls.append([url, title, timestamp])
        conn.close()
    except Exception as e:
        print("Error reading Chrome history:", e)
    finally:
        if os.path.exists(temp_history):
            os.remove(temp_history)
    return urls

def get_firefox_history():
    firefox_path = os.path.expanduser(
        "~\\AppData\\Roaming\\Mozilla\\Firefox\\Profiles"
    )
    
    profiles = [f for f in os.listdir(firefox_path) if f.endswith(".default") or f.endswith(".default-release")]
    urls = []

    for profile in profiles:
        history_file = os.path.join(firefox_path, profile, "places.sqlite")
        if os.path.exists(history_file):
            temp_history = "places_temp.sqlite"
            try:
                shutil.copy2(history_file, temp_history)
            except Exception as e:
                print("Error copying Firefox history:", e)
                continue

            try:
                conn = sqlite3.connect(temp_history)
                cursor = conn.cursor()
                cursor.execute("SELECT url, title, last_visit_date FROM moz_places")
                for url, title, last_visit_date in cursor.fetchall():
                    if last_visit_date:
                        timestamp = datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=last_visit_date)
                    else:
                        timestamp = ""
                    urls.append([url, title, timestamp])
                conn.close()
            except Exception as e:
                print("Error reading Firefox history:", e)
            finally:
                if os.path.exists(temp_history):
                    os.remove(temp_history)
    return urls

def save_to_csv(data, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["URL", "Title", "Last Visit"])
        writer.writerows(data)
    print(f"Saved history to {filename}")

if __name__ == "__main__":
    print("Collecting Chrome history...")
    chrome_data = get_chrome_history()
    print(f"Found {len(chrome_data)} Chrome records.")

    print("Collecting Firefox history...")
    firefox_data = get_firefox_history()
    print(f"Found {len(firefox_data)} Firefox records.")

    all_data = chrome_data + firefox_data
    save_to_csv(all_data, "browser_history.csv")
