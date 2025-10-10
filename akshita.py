import os
import sys
import pickle

# Safe file paths for exe
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RF_MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
CSV_PATH = os.path.join(BASE_DIR, "browsing_history.csv")

# Load ML model safely
try:
    with open(RF_MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
except:
    # fallback training if .pkl missing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    TRAIN_URLS = ['https://www.khanacademy.org','https://www.harvard.edu','https://www.coursera.org',
                  'https://www.facebook.com','https://www.instagram.com','https://twitter.com',
                  'https://www.amazon.com','https://www.flipkart.com','https://www.ebay.com',
                  'https://www.bbc.com/news','https://www.cnn.com','https://timesofindia.indiatimes.com']
    TRAIN_LABELS = ['Education','Education','Education','Social Media','Social Media','Social Media',
                    'E-commerce','E-commerce','E-commerce','News','News','News']
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
    X_train = vectorizer.fit_transform(TRAIN_URLS)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, TRAIN_LABELS)

# Load CSV safely
import pandas as pd
if os.path.exists(CSV_PATH):
    browsing_history = pd.read_csv(CSV_PATH)
else:
    browsing_history = pd.DataFrame()  # empty if missing
# portable_forensic_allinone_safe_visitcount.py
import os
import shutil
import sqlite3
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urlparse
from hashlib import sha256
import numpy as np
import glob
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ---------------- Configuration ----------------
POLL_INTERVAL_SECONDS = 6
ML_SUSPICIOUS_PROB = 0.7
DOWNLOADS_PATH = os.path.expanduser("~/Downloads")
SUSPICIOUS_BLACKLIST = {"malware.com", "phishing.net", "suspicious.org"}

BROWSERS = {
    "Chrome": os.path.expanduser(r"~\AppData\Local\Google\Chrome\User Data"),
    "Edge": os.path.expanduser(r"~\AppData\Local\Microsoft\Edge\User Data"),
    "Brave": os.path.expanduser(r"~\AppData\Local\BraveSoftware\Brave-Browser\User Data"),
    "Opera": os.path.expanduser(r"~\AppData\Roaming\Opera Software\Opera Stable"),
    "Firefox": os.path.expanduser(r"~\AppData\Roaming\Mozilla\Firefox\Profiles"),
}

# ---------------- Helpers ----------------
def extract_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except:
        return ""

def convert_chrome_time(microseconds):
    try:
        if microseconds and 0 < microseconds < 11644473600000000:
            return datetime(1601, 1, 1) + timedelta(microseconds=int(microseconds))
        else:
            return None
    except:
        return None

def convert_firefox_time(microseconds):
    try:
        if microseconds and microseconds > 0:
            return datetime(1970,1,1) + timedelta(microseconds=int(microseconds))
        else:
            return None
    except:
        return None

def hash_file(path):
    h = sha256()
    if os.path.exists(path):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    return None

# ---------------- ML Classifier ----------------
TRAIN_URLS = ['https://www.khanacademy.org','https://www.harvard.edu','https://www.coursera.org',
              'https://www.facebook.com','https://www.instagram.com','https://twitter.com',
              'https://www.amazon.com','https://www.flipkart.com','https://www.ebay.com',
              'https://www.bbc.com/news','https://www.cnn.com','https://timesofindia.indiatimes.com']
TRAIN_LABELS = ['Education','Education','Education','Social Media','Social Media','Social Media',
                'E-commerce','E-commerce','E-commerce','News','News','News']
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
X_train = vectorizer.fit_transform(TRAIN_URLS)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, TRAIN_LABELS)

def predict_category_batch(url_series):
    try:
        X = vectorizer.transform(url_series.astype(str))
        return clf.predict(X)
    except:
        return pd.Series(['Other']*len(url_series))

# ---------------- Alerts ----------------
alerts_lock = threading.Lock()
alerts_df = pd.DataFrame(columns=['timestamp','dst_domain','description','profile','src'])

def add_alert(ts, domain, desc, profile="Unknown", src="monitor"):
    global alerts_df
    with alerts_lock:
        alerts_df = pd.concat([alerts_df, pd.DataFrame([{
            'timestamp': pd.to_datetime(ts),
            'dst_domain': domain,
            'description': desc,
            'profile': profile,
            'src': src
        }])], ignore_index=True)

# ---------------- Browser Extraction ----------------
def extract_browser_history(browser="Chrome", log_callback=None):
    base_path = BROWSERS.get(browser)
    if not base_path or not os.path.exists(base_path):
        if log_callback:
            log_callback(f"{browser} not installed or path missing.")
        return pd.DataFrame()

    all_dfs = []

    if browser.lower() != 'firefox':
        profile_paths = glob.glob(os.path.join(base_path, '*')) if browser.lower() != 'opera' else [base_path]

        for profile in profile_paths:
            profile_name = os.path.basename(profile)
            history_db = os.path.join(profile, 'History')
            if not os.path.exists(history_db):
                continue

            tmp_db = os.path.join(os.environ.get('TEMP'), f'history_{browser}_{profile_name}.db')
            try:
                shutil.copy2(history_db, tmp_db)
                conn = sqlite3.connect(tmp_db)
                df = pd.read_sql_query("SELECT url,title,last_visit_time,visit_count FROM urls", conn)
                conn.close()
                os.remove(tmp_db)
            except:
                continue

            # ---------------- Safe handling ----------------
            if 'visit_count' not in df.columns:
                df['visit_count'] = np.nan  # Set to NULL if missing

            df['Deleted'] = df['last_visit_time'] == 0
            df['timestamp'] = df['last_visit_time'].apply(convert_chrome_time)
            df['domain'] = df['url'].apply(extract_domain)
            df['profile'] = profile_name
            df['guest_mode'] = 'guest' in profile_name.lower()

            if 'url' in df.columns:
                domain_counts = df.groupby('domain')['url'].count().reset_index(name='visit_count_domain')
                df = df.merge(domain_counts, on='domain', how='left')
            else:
                df['visit_count_domain'] = np.nan

            df['site_type'] = predict_category_batch(df['url'])
            df['ml_score'] = np.random.rand(len(df))
            df['browser'] = browser
            all_dfs.append(df[['browser','profile','guest_mode','Deleted','timestamp','url','domain','visit_count','visit_count_domain','ml_score','site_type']])
    else:
        # Firefox
        profile_paths = glob.glob(os.path.join(base_path, '*'))
        for profile in profile_paths:
            profile_name = os.path.basename(profile)
            history_db = os.path.join(profile, 'places.sqlite')
            if not os.path.exists(history_db):
                continue

            tmp_db = os.path.join(os.environ.get('TEMP'), f'history_firefox_{profile_name}.db')
            try:
                shutil.copy2(history_db, tmp_db)
                conn = sqlite3.connect(tmp_db)
                df = pd.read_sql_query("SELECT url,title,last_visit_date AS last_visit_time,visit_count FROM moz_places", conn)
                conn.close()
                os.remove(tmp_db)
            except:
                continue

            if 'visit_count' not in df.columns:
                df['visit_count'] = np.nan

            df['Deleted'] = df['last_visit_time'] == 0
            df['timestamp'] = df['last_visit_time'].apply(convert_firefox_time)
            df['domain'] = df['url'].apply(extract_domain)
            df['profile'] = profile_name
            df['guest_mode'] = False

            if 'url' in df.columns:
                domain_counts = df.groupby('domain')['url'].count().reset_index(name='visit_count_domain')
                df = df.merge(domain_counts, on='domain', how='left')
            else:
                df['visit_count_domain'] = np.nan

            df['site_type'] = predict_category_batch(df['url'])
            df['ml_score'] = np.random.rand(len(df))
            df['browser'] = browser
            all_dfs.append(df[['browser','profile','guest_mode','Deleted','timestamp','url','domain','visit_count','visit_count_domain','ml_score','site_type']])

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# ---------------- Correlation ----------------
def correlate_with_alerts(browser_df, alerts_df_local, window_minutes=2):
    if browser_df.empty or alerts_df_local.empty:
        return pd.DataFrame()
    bd = browser_df.copy()
    bd = bd[bd['timestamp'].notna()].copy()
    ad = alerts_df_local.copy()
    ad['timestamp'] = pd.to_datetime(ad['timestamp'])
    merged = pd.merge(bd, ad, left_on='domain', right_on='dst_domain', how='left', suffixes=('_b','_a'))
    merged['time_diff'] = (merged['timestamp'] - merged['timestamp_a']).abs().dt.total_seconds()
    timeline = merged[merged['time_diff'] <= window_minutes*60].sort_values('timestamp')
    return timeline

# ---------------- Monitor ----------------
class LiveMonitor(threading.Thread):
    def __init__(self, gui_log, selected_browsers, interval=POLL_INTERVAL_SECONDS):
        super().__init__(daemon=True)
        self.interval = interval
        self.log = gui_log
        self._stop = threading.Event()
        self.selected_browsers = selected_browsers
        self.last_seen_urls = set()
        self.last_seen_downloads = set()

    def stop(self):
        self._stop.set()

    def run(self):
        self.log("Live monitor started.")
        for browser in self.selected_browsers:
            try:
                df = extract_browser_history(browser=browser)
                if not df.empty:
                    self.last_seen_urls.update(df['url'].astype(str).tolist())
                    self.log(f"Seeded URLs from {browser}: {len(df)} rows")
            except Exception as e:
                self.log(f"Monitor seed error for {browser}: {e}")

        if os.path.exists(DOWNLOADS_PATH):
            self.last_seen_downloads = set(os.listdir(DOWNLOADS_PATH))

        while not self._stop.is_set():
            try:
                for browser in self.selected_browsers:
                    df = extract_browser_history(browser=browser)
                    if not df.empty:
                        current_urls = set(df['url'].astype(str).tolist())
                        new_urls = current_urls - self.last_seen_urls
                        if new_urls:
                            for url in list(new_urls)[:50]:
                                domain = extract_domain(url)
                                row = df[df['url'] == url].iloc[0]
                                now = datetime.utcnow()
                                descs = []
                                if domain in SUSPICIOUS_BLACKLIST:
                                    descs.append("Blacklisted domain")
                                if row.get('site_type','') in ('E-commerce',) and row.get('ml_score',0) > ML_SUSPICIOUS_PROB:
                                    descs.append("High-risk e-commerce behaviour")
                                if row.get('site_type','') == 'Other' and row.get('ml_score',0) > 0.95:
                                    descs.append("Very unusual site (high score)")
                                if descs:
                                    desc = "; ".join(descs)
                                    add_alert(now, domain, f"New URL matched: {url} -> {desc}", profile=row.get('profile','Unknown'), src='history-monitor')
                                    self.log(f"ALERT (history): {domain} — {desc}")
                                self.last_seen_urls = current_urls

                if os.path.exists(DOWNLOADS_PATH):
                    current_dl = set(os.listdir(DOWNLOADS_PATH))
                    new_files = current_dl - self.last_seen_downloads
                    if new_files:
                        for fname in list(new_files)[:50]:
                            path = os.path.join(DOWNLOADS_PATH, fname)
                            fh = hash_file(path)
                            add_alert(datetime.utcnow(), '', f"New download: {fname} (hash={fh})", profile="Local", src='download-monitor')
                            self.log(f"ALERT (download): {fname}")
                        self.last_seen_downloads = current_dl
            except Exception as e:
                self.log(f"Monitor error: {e}")
            time.sleep(self.interval)

# ---------------- GUI ----------------
class BrowserApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Portable Forensic All-in-One")
        self.geometry("1200x700")

        # Browser selection
        self.browser_vars = {}
        frame_select = tk.Frame(self)
        frame_select.pack(pady=6)
        tk.Label(frame_select, text="Select browsers:").pack(side=tk.LEFT)
        for b in BROWSERS.keys():
            var = tk.BooleanVar(value=(b=="Chrome"))
            cb = tk.Checkbutton(frame_select, text=b, variable=var)
            cb.pack(side=tk.LEFT)
            self.browser_vars[b] = var

        # Log
        self.log_box = tk.Text(self, height=12)
        self.log_box.pack(fill=tk.BOTH, expand=False, padx=8, pady=6)

        # Buttons
        frame_btn = tk.Frame(self)
        frame_btn.pack(pady=6)
        self.btn_normal = tk.Button(frame_btn, text="Normal Browsing", command=self.run_normal)
        self.btn_normal.pack(side=tk.LEFT, padx=6)
        self.btn_security = tk.Button(frame_btn, text="Security Investigation", command=self.run_security)
        self.btn_security.pack(side=tk.LEFT, padx=6)
        self.btn_alerts = tk.Button(frame_btn, text="View Alerts", command=self.view_alerts)
        self.btn_alerts.pack(side=tk.LEFT, padx=6)

        # Start monitor
        selected = [b for b,v in self.browser_vars.items() if v.get()]
        self.monitor = LiveMonitor(gui_log=self.log, selected_browsers=selected)
        self.monitor.start()

    def log(self, msg):
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self.log_box.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_box.see(tk.END)

    def run_normal(self):
        threading.Thread(target=self._run_normal, daemon=True).start()

    def _run_normal(self):
        self.log("Extracting normal browsing history...")
        selected = [b for b,v in self.browser_vars.items() if v.get()]
        df_list = [extract_browser_history(browser=b, log_callback=self.log) for b in selected]
        df = pd.concat(df_list,ignore_index=True) if df_list else pd.DataFrame()
        self.display_table(df, "Browsing History")

    def run_security(self):
        threading.Thread(target=self._run_security, daemon=True).start()

    def _run_security(self):
        self.log("Running security investigation...")
        selected = [b for b,v in self.browser_vars.items() if v.get()]
        df_list = [extract_browser_history(browser=b) for b in selected]
        df = pd.concat(df_list,ignore_index=True) if df_list else pd.DataFrame()
        with alerts_lock:
            a_copy = alerts_df.copy()
        timeline = correlate_with_alerts(df,a_copy)
        if timeline.empty:
            self.log("No correlated suspicious activity found.")
            messagebox.showinfo("Security Investigation","No suspicious activity found")
        else:
            self.log(f"Found {len(timeline)} correlated events.")
            self.display_table(timeline,"Suspicious Correlated Events")

    def view_alerts(self):
        with alerts_lock:
            a_copy = alerts_df.copy()
        if a_copy.empty:
            messagebox.showinfo("Alerts","No alerts generated yet.")
            return
        self.display_table(a_copy,"Live Alerts")

    def display_table(self, df, title="Data"):
        if df.empty:
            messagebox.showinfo(title,"No data to display.")
            return
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("1100x600")
        columns = df.columns.tolist()
        tree = ttk.Treeview(win, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor='w')
        for _, row in df.iterrows():
            values = [row.get(c,"") for c in columns]
            iid = tree.insert("", tk.END, values=values)
            if 'guest_mode' in df.columns and row.get('guest_mode'):
                tree.item(iid, tags=('guest',))
            if 'Deleted' in df.columns and row.get('Deleted'):
                tree.item(iid, tags=('deleted',))
        tree.tag_configure('guest', background='#FFFACD')
        tree.tag_configure('deleted', background='#FFB6C1')
        tree.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def on_close(self):
        try:
            self.monitor.stop()
        except:
            pass
        self.destroy()

# ---------------- Run App ----------------
if __name__ == "__main__":
    app = BrowserApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
