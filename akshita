import os
import tempfile
import shutil
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import scipy.sparse
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# ===== Feature engineering & rules =====
def extract_url_features(url):
    u = url.lower()
    length = len(u)
    digits = sum(c.isdigit() for c in u)
    digit_ratio = digits / max(1, length)
    count_dash = u.count('-')
    count_q = u.count('?')
    count_slash = u.count('/')
    count_eq = u.count('=')
    has_ip = 1 if any(part.isdigit() and '.' in part for part in u.split('/')[:1]) else 0
    tld_com = 1 if ".com" in u else 0
    tld_org = 1 if ".org" in u else 0
    tld_net = 1 if ".net" in u else 0
    return [length, digits, round(digit_ratio,3), count_dash, count_q, count_slash, count_eq, has_ip, tld_com, tld_org, tld_net]

def is_suspicious_rule(url):
    u = url.lower()
    suspicious_keywords = ["phish", "malware", "login", "verify", "secure", "update", "bank", "confirm"]
    if any(k in u for k in suspicious_keywords):
        return True, "Contains suspicious keyword"
    if sum(c.isdigit() for c in u) > 8 and len(u) > 40:
        return True, "Too many digits / long URL"
    if ".." in u or "@@" in u:
        return True, "Obfuscated pattern"
    if u.count('-') > 6:
        return True, "Many dashes"
    return False, ""

# ===== Browser history extraction =====
def read_normal_history(db_path):
    urls = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT url, title, visit_count, last_visit_time FROM urls ORDER BY last_visit_time DESC LIMIT 200")
        rows = cursor.fetchall()
        for url, title, count, last_visit in rows:
            urls.append((url, title, count, last_visit))
    except:
        pass
    finally:
        try: conn.close()
        except: pass
    return urls

def recover_deleted_records(db_path):
    recovered = set()
    try:
        with open(db_path, "rb") as f:
            data = f.read()
        page_size = int.from_bytes(data[16:18], byteorder='big')
        if page_size == 1:
            page_size = 65536
        for i in range(0, len(data), page_size):
            page = data[i:i+page_size]
            if page[0] in (0x0D, 0x00):
                try:
                    text = page.decode('utf-8', errors='ignore')
                    for line in text.split('\x00'):
                        if line.startswith("http") or line.startswith("www"):
                            recovered.add(line.strip())
                except:
                    continue
    except:
        pass
    return list(recovered)

# ===== Option B: training ML model =====
def train_optionB_model():
    train_urls = [
        "youtube.com", "youtube.com/watch?v=abc", "m.youtube.com/shorts/xxx",
        "wikipedia.org", "en.wikipedia.org/wiki/Machine_learning",
        "facebook.com", "instagram.com", "twitter.com",
        "coursera.org", "khanacademy.org", "edx.org", "stackoverflow.com",
        "onlinegdb.com", "repl.it", "ideone.com",
        "google.com/search?q=java+online+compiler",
        "abc-phish.com/login", "secure-login-abc123.com", "xyz-malware.net/download",
        "bank-update-verify.com/login", "login-secure.accounts-verify.net"
    ]  # 21 items

    train_labels = [
        "Entertainment","Entertainment","Entertainment",
        "Education","Education",
        "Social","Social","Social",
        "Education","Education","Education","Education",
        "Education","Education","Education",
        "Education",
        "Malicious","Malicious","Malicious",
        "Malicious","Malicious"
    ]  # 21 items

    assert len(train_urls) == len(train_labels), "URLs and labels must match!"

    numeric_features = [extract_url_features(u) for u in train_urls]
    X_numeric = scipy.sparse.csr_matrix(np.array(numeric_features))
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,6))
    X_tfidf = tfidf.fit_transform(train_urls)
    X_final = scipy.sparse.hstack([X_tfidf, X_numeric], format='csr')
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    clf.fit(X_final, train_labels)
    return clf, tfidf

# ===== Main GUI =====
def main_gui():
    root = tk.Tk()
    root.title("Browser History Project - Bulk ML")
    root.geometry("1050x650")

    center_frame = tk.Frame(root)
    center_frame.pack(pady=10)

    scan_btn = tk.Button(center_frame, text="Scan & Classify", width=25)
    scan_btn.pack(pady=5)

    loading_var = tk.StringVar(value="")
    loading_label = tk.Label(center_frame, textvariable=loading_var, fg="blue")
    loading_label.pack(pady=5)

    cols = ("profile", "url", "title", "visits", "category", "safety", "context")
    tree = ttk.Treeview(root, columns=cols, show='headings', height=24)
    for c in cols:
        tree.heading(c, text=c.title())
        tree.column(c, width=140 if c=='url' else 100, anchor='w')
    tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    status = tk.StringVar(value="Ready")
    tk.Label(root, textvariable=status).pack(fill=tk.X, padx=8, pady=4)

    def threaded_scan():
        try:
            loading_var.set("⏳ Loading... please wait")
            status.set("Scanning Chrome history...")
            scan_btn.config(state="disabled")
            root.update_idletasks()

            chrome_user_data = os.path.join(os.environ["USERPROFILE"], "AppData", "Local", "Google", "Chrome", "User Data")
            profiles = []
            if os.path.exists(os.path.join(chrome_user_data, "Default")):
                profiles.append("Default")
            for folder in os.listdir(chrome_user_data):
                if folder.startswith("Profile") or folder == "Guest Profile":
                    profiles.append(folder)

            all_normal_history = []
            all_deleted_history = []

            for profile in profiles:
                history_db = os.path.join(chrome_user_data, profile, "History")
                if os.path.exists(history_db):
                    temp_path = os.path.join(tempfile.gettempdir(), f"History_copy_{profile}")
                    shutil.copy2(history_db, temp_path)
                    normal_history = [(url, title, count, last_visit, profile, "Guest" if profile=="Guest Profile" else "Normal")
                                      for url, title, count, last_visit in read_normal_history(temp_path)]
                    deleted_history = [(url, profile, "Deleted") for url in recover_deleted_records(temp_path)]
                    all_normal_history += normal_history
                    all_deleted_history += deleted_history
                    try: os.remove(temp_path)
                    except: pass

            df_live = pd.DataFrame(all_normal_history, columns=["url", "title", "visit_count", "last_visit_time", "profile", "context"])
            df_deleted = pd.DataFrame(all_deleted_history, columns=["url", "profile", "context"])
            if not df_live.empty:
                status.set("Training ML and predicting in bulk...")
                root.update_idletasks()

                clf, tfidf = train_optionB_model()

                urls = df_live['url'].tolist()
                numeric_features = np.array([extract_url_features(u) for u in urls])
                X_numeric = scipy.sparse.csr_matrix(numeric_features)
                X_urls = tfidf.transform(urls)
                X_final = scipy.sparse.hstack([X_urls, X_numeric], format='csr')
                pred_categories = clf.predict(X_final)

                safety_flags = []
                for u, cat in zip(urls, pred_categories):
                    suspicious_flag, _ = is_suspicious_rule(u)
                    safety_flags.append("Malicious" if suspicious_flag or cat=="Malicious" else "Safe")

                for r in tree.get_children():
                    tree.delete(r)
                for i, row in df_live.iterrows():
                    tree.insert("", "end", values=(
                        row['profile'],
                        row['url'],
                        row['title'][:60],
                        row['visit_count'],
                        pred_categories[i],
                        safety_flags[i],
                        row['context']
                    ))
                # Add deleted URLs
                for i, row in df_deleted.iterrows():
                    tree.insert("", "end", values=(
                        row['profile'],
                        row['url'],
                        "",
                        0,
                        "Malicious",
                        "Malicious",
                        row['context']
                    ))

                status.set(f"Scan completed — {len(df_live)+len(df_deleted)} records processed")
            else:
                status.set("No history found for current user.")
                messagebox.showinfo("No data", "No Chrome history found.")

        finally:
            loading_var.set("")
            scan_btn.config(state="normal")

    scan_btn.config(command=lambda: threading.Thread(target=threaded_scan, daemon=True).start())
    root.mainloop()

if __name__ == "__main__":
    main_gui()
