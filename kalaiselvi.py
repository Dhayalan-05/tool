import os
import sys
import tempfile
import shutil
import sqlite3
import struct
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import scipy
import tkinter as tk
from tkinter import scrolledtext
import joblib

# ===== Helper to locate resources in EXE =====
def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ========== PERSON A ==========
def extract_search_terms(history_db_path):
    search_terms = []
    conn = sqlite3.connect(history_db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT term, url FROM keyword_search_terms JOIN urls ON keyword_search_terms.url_id = urls.id")
        results = cursor.fetchall()
        for term, url in results:
            search_terms.append((term, url))
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return search_terms

# ========== PERSON B ==========
def read_normal_history(db_path):
    urls = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT url, title, visit_count, last_visit_time FROM urls ORDER BY last_visit_time DESC LIMIT 50")
        rows = cursor.fetchall()
        for url, title, count, last_visit in rows:
            urls.append((url, title, count, last_visit))
    except Exception:
        pass
    finally:
        conn.close()
    return urls

def recover_deleted_records(db_path):
    recovered = set()
    with open(db_path, "rb") as f:
        data = f.read()

    page_size = struct.unpack(">H", data[16:18])[0]
    if page_size == 1:
        page_size = 65536

    for i in range(0, len(data), page_size):
        page = data[i:i+page_size]
        if page[0] == 0x0D or page[0] == 0x00:
            try:
                text = page.decode('utf-8', errors='ignore')
                for line in text.split('\x00'):
                    if line.startswith("http") or line.startswith("www"):
                        recovered.add(line.strip())
            except Exception:
                continue
    return list(recovered)

# ========== PERSON C ==========
def train_or_load_ml_model(csv_path):
    model_file = resource_path("rf_model.pkl")
    vector_file = resource_path("tfidf_vectorizer.pkl")

    if os.path.exists(model_file) and os.path.exists(vector_file):
        clf = joblib.load(model_file)
        tfidf = joblib.load(vector_file)
    else:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)
        X = df[["url", "hour", "duration"]]
        y = df["category"]

        tfidf = TfidfVectorizer()
        X_url = tfidf.fit_transform(X["url"])
        X_numeric = X[["hour", "duration"]].values
        X_final = scipy.sparse.hstack([X_url, X_numeric])

        X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Save model
        joblib.dump(clf, model_file)
        joblib.dump(tfidf, vector_file)

    return clf, tfidf

# ========== PERSON D (YOU) ==========
def main_gui():
    root = tk.Tk()
    root.title("Browser History Project Output")
    root.geometry("950x650")

    output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=120, height=35)
    output_box.pack(padx=10, pady=10)

    output_box.insert(tk.END, "\n🚀 Starting Combined Browser History Project...\n\n")

    chrome_user_data = os.path.join(os.environ["USERPROFILE"], "AppData", "Local", "Google", "Chrome", "User Data")

    # Detect all profiles
    profiles = []
    if os.path.exists(os.path.join(chrome_user_data, "Default")):
        profiles.append("Default")
    for folder in os.listdir(chrome_user_data):
        if folder.startswith("Profile"):
            profiles.append(folder)

    all_search_terms = []
    all_normal_history = []
    all_deleted_history = []

    for profile in profiles:
        profile_path = os.path.join(chrome_user_data, profile, "History")
        if os.path.exists(profile_path):
            temp_path = os.path.join(tempfile.gettempdir(), f"History_copy_{profile}")
            shutil.copy2(profile_path, temp_path)

            all_search_terms += extract_search_terms(temp_path)
            all_normal_history += read_normal_history(temp_path)
            all_deleted_history += recover_deleted_records(temp_path)

            os.remove(temp_path)

    # Assign to main variables
    search_terms = all_search_terms
    normal_history = all_normal_history
    deleted_history = all_deleted_history

    # Train/load ML model
    csv_file = resource_path("browsing_history.csv")
    clf = tfidf = None
    if os.path.exists(csv_file):
        clf, tfidf = train_or_load_ml_model(csv_file)

    # Predict categories for live history if ML available
    predictions = []
    if clf and tfidf:
        df_live = pd.DataFrame(normal_history, columns=["url", "title", "visit_count", "last_visit_time"])
        df_live["hour"] = pd.to_datetime(df_live["last_visit_time"]).dt.hour
        df_live["duration"] = df_live["visit_count"]  # simple proxy

        X_new = tfidf.transform(df_live["url"])
        preds = clf.predict(X_new)
        df_live["predicted_category"] = preds
        predictions = df_live[["url", "predicted_category"]].values.tolist()

    # ===== Display in GUI =====
    output_box.insert(tk.END, "===== FINAL REPORT =====\n\n")

    output_box.insert(tk.END, "🔹 Search Terms:\n")
    for term, url in search_terms[:10]:
        output_box.insert(tk.END, f" - {term} → {url}\n")
    output_box.insert(tk.END, "\n")

    output_box.insert(tk.END, "🔹 Browsing History (recent):\n")
    for url, title, count, last_visit in normal_history[:20]:
        output_box.insert(tk.END, f" - {url} | Title: {title} | Visits: {count}\n")
    output_box.insert(tk.END, "\n")

    output_box.insert(tk.END, "🔹 Recovered Deleted URLs:\n")
    for url in deleted_history[:10]:
        output_box.insert(tk.END, f" - {url}\n")
    output_box.insert(tk.END, "\n")

    if predictions:
        output_box.insert(tk.END, "🔹 ML Predictions for Live History:\n")
        for url, category in predictions[:20]:
            output_box.insert(tk.END, f" - {url} → {category}\n")
        output_box.insert(tk.END, "\n")

    output_box.insert(tk.END, "===== END OF REPORT =====\n")
    root.mainloop()


if __name__ == "__main__":
    main_gui()
