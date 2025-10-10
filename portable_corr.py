# portable_forensic_allinone.py
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

# Optional ML libs - required for classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ----------------- Configuration -----------------
POLL_INTERVAL_SECONDS = 6          # monitor polling interval
RECENT_WINDOW_SECONDS = 300        # consider an entry "recent" if within this many seconds
ML_SUSPICIOUS_PROB = 0.7           # placeholder threshold (not used for RandomForest predict_proba here)
DOWNLOADS_PATH = os.path.expanduser("~/Downloads")
CHROME_USERDATA = os.path.expanduser(r"~\AppData\Local\Google\Chrome\User Data")
SUSPICIOUS_BLACKLIST = {"malware.com", "phishing.net", "suspicious.org"}

# ----------------- Helpers -----------------
def extract_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except:
        return ""

def convert_chrome_time(microseconds):
    try:
        if microseconds and microseconds > 0 and microseconds < 11644473600000000:
            return datetime(1601, 1, 1) + timedelta(microseconds=int(microseconds))
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

# ----------------- Small built-in ML classifier -----------------
TRAIN_URLS = [
    'https://www.khanacademy.org', 'https://www.harvard.edu', 'https://www.coursera.org',
    'https://www.facebook.com', 'https://www.instagram.com', 'https://twitter.com',
    'https://www.amazon.com', 'https://www.flipkart.com', 'https://www.ebay.com',
    'https://www.bbc.com/news', 'https://www.cnn.com', 'https://timesofindia.indiatimes.com'
]
TRAIN_LABELS = [
    'Education', 'Education', 'Education',
    'Social Media', 'Social Media', 'Social Media',
    'E-commerce', 'E-commerce', 'E-commerce',
    'News', 'News', 'News'
]
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
X_train = vectorizer.fit_transform(TRAIN_URLS)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, TRAIN_LABELS)

def predict_category(url):
    try:
        x = vectorizer.transform([str(url)])
        return clf.predict(x)[0]
    except:
        return "Other"

# ----------------- In-memory alerts store -----------------
# columns: timestamp (datetime), dst_domain, description, profile, src (monitor type)
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

# ----------------- Browser history extraction -----------------
def extract_browser_history(log_callback=None):
    base_path = CHROME_USERDATA
    profile_paths = glob.glob(os.path.join(base_path, "*"))
    all_dfs = []
    if log_callback:
        log_callback(f"Scanning profiles in: {base_path}")
    for profile in profile_paths:
        profile_name = os.path.basename(profile)
        history_db = os.path.join(profile, "History")
        if not os.path.exists(history_db):
            if log_callback:
                log_callback(f"Skipping (no History): {profile_name}")
            continue
        tmp_db = os.path.join(os.environ.get("TEMP"), f"history_{profile_name}.db")
        try:
            shutil.copy2(history_db, tmp_db)
        except Exception as e:
            if log_callback:
                log_callback(f"Copy error for {profile_name}: {e}")
            continue
        try:
            conn = sqlite3.connect(tmp_db)
            query = "SELECT url, title, last_visit_time, visit_count FROM urls"
            df = pd.read_sql_query(query, conn)
        except Exception as e:
            if log_callback:
                log_callback(f"Read error {profile_name}: {e}")
            try:
                conn.close()
            except:
                pass
            try:
                os.remove(tmp_db)
            except:
                pass
            continue
        conn.close()
        try:
            os.remove(tmp_db)
        except:
            pass

        # Convert and annotate
        df['Deleted'] = df['last_visit_time'].apply(lambda x: True if x == 0 else False)
        df['timestamp'] = df['last_visit_time'].apply(convert_chrome_time)
        df['domain'] = df['url'].apply(extract_domain)
        df['profile'] = profile_name
        df['guest_mode'] = 'guest' in profile_name.lower() or 'guest profile' in profile_name.lower()

        # visit_count per domain
        domain_counts = df.groupby('domain')['url'].count().reset_index(name='visit_count')
        df = df.merge(domain_counts, on='domain', how='left', suffixes=('','_domain'))

        # classify
        df['site_type'] = df['domain'].apply(predict_category)
        # placeholder ml_score (use real model later if available)
        df['ml_score'] = np.random.rand(len(df))

        # pick desired columns
        all_dfs.append(df[['profile','guest_mode','Deleted','timestamp','url','domain','visit_count','ml_score','site_type']])
        if log_callback:
            log_callback(f"Read {len(df)} rows from {profile_name}")
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('timestamp', ascending=False).reset_index(drop=True)
        return combined
    return pd.DataFrame()

# ----------------- Correlation (uses in-memory alerts instead of external CSV) -----------------
def correlate_with_alerts(browser_df, alerts_df_local, window_minutes=2):
    """Return browser rows that match alerts by domain and time window."""
    if browser_df.empty or alerts_df_local.empty:
        return pd.DataFrame()
    # ensure timestamp types
    bd = browser_df.copy()
    bd = bd[bd['timestamp'].notna()].copy()
    ad = alerts_df_local.copy()
    ad['timestamp'] = pd.to_datetime(ad['timestamp'])

    merged = pd.merge(bd, ad, left_on='domain', right_on='dst_domain', how='left', suffixes=('_b','_a'))
    merged['time_diff'] = (merged['timestamp'] - merged['timestamp_a']).abs().dt.total_seconds()
    timeline = merged[merged['time_diff'] <= window_minutes*60].sort_values('timestamp')
    return timeline

# ----------------- Monitor: polls history & downloads -----------------
class LiveMonitor(threading.Thread):
    def __init__(self, gui_log, interval=POLL_INTERVAL_SECONDS):
        super().__init__(daemon=True)
        self.interval = interval
        self.log = gui_log
        self._stop = threading.Event()
        self.last_seen_urls = set()
        self.last_seen_downloads = set()

    def stop(self):
        self._stop.set()

    def run(self):
        self.log("Live monitor started.")
        # seed last_seen from current history + downloads
        try:
            df = extract_browser_history(log_callback=self.log)
            if not df.empty:
                self.last_seen_urls = set(df['url'].astype(str).tolist())
        except Exception as e:
            self.log(f"Monitor seed history error: {e}")

        try:
            if os.path.exists(DOWNLOADS_PATH):
                self.last_seen_downloads = set(os.listdir(DOWNLOADS_PATH))
        except Exception as e:
            self.log(f"Monitor seed downloads error: {e}")

        while not self._stop.is_set():
            try:
                df = extract_browser_history(log_callback=None)
                if not df.empty:
                    # check for new URLs
                    current_urls = set(df['url'].astype(str).tolist())
                    new_urls = current_urls - self.last_seen_urls
                    if new_urls:
                        for url in list(new_urls)[:50]:  # cap to avoid flooding
                            domain = extract_domain(url)
                            row = df[df['url'] == url].iloc[0]
                            now = datetime.utcnow()
                            # suspicious if blacklist or certain site types
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
                                # notify GUI
                                self.log(f"ALERT (history): {domain} — {desc}")
                                # popup on GUI thread
                                try:
                                    self._popup_alert(domain, desc, source='history')
                                except:
                                    pass
                            # even if not suspicious, still add lightweight monitoring alert for certain categories
                            if domain in SUSPICIOUS_BLACKLIST and not descs:
                                add_alert(now, domain, f"Visited blacklisted domain {domain}", profile=row.get('profile','Unknown'), src='history-monitor')
                                self.log(f"ALERT (blacklist): {domain}")
                        self.last_seen_urls = current_urls

                # check downloads folder for new files
                if os.path.exists(DOWNLOADS_PATH):
                    current_dl = set(os.listdir(DOWNLOADS_PATH))
                    new_files = current_dl - self.last_seen_downloads
                    if new_files:
                        for fname in list(new_files)[:50]:
                            path = os.path.join(DOWNLOADS_PATH, fname)
                            fh = hash_file(path)
                            add_alert(datetime.utcnow(), '', f"New download: {fname} (hash={fh})", profile="Local", src='download-monitor')
                            self.log(f"ALERT (download): {fname}")
                            self._popup_alert(fname, "New download detected", source='download')
                        self.last_seen_downloads = current_dl

            except Exception as e:
                self.log(f"Monitor error: {e}")

            time.sleep(self.interval)

    def _popup_alert(self, title, text, source='monitor'):
        # show popup from GUI thread - schedule with after() if possible
        try:
            # we assume there's a global app variable (created later)
            app.after(1, lambda: messagebox.showwarning(f"Security Alert ({source})", f"{title}\n\n{text}"))
        except Exception:
            pass

# ----------------- GUI Application -----------------
class BrowserApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Portable Forensic All-in-One")
        self.geometry("1200x700")

        # Log area
        self.log_box = tk.Text(self, height=12)
        self.log_box.pack(fill=tk.BOTH, expand=False, padx=8, pady=6)

        # Buttons
        frame = tk.Frame(self)
        frame.pack(pady=6)
        self.btn_normal = tk.Button(frame, text="Normal Browsing", command=self.run_normal)
        self.btn_normal.pack(side=tk.LEFT, padx=6)
        self.btn_security = tk.Button(frame, text="Security Investigation", command=self.run_security)
        self.btn_security.pack(side=tk.LEFT, padx=6)
        self.btn_alerts = tk.Button(frame, text="View Alerts", command=self.view_alerts)
        self.btn_alerts.pack(side=tk.LEFT, padx=6)

        # Start the live monitor thread
        self.monitor = LiveMonitor(gui_log=self.log)
        self.monitor.start()

    def log(self, msg):
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self.log_box.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_box.see(tk.END)

    # Normal browsing display
    def run_normal(self):
        threading.Thread(target=self._run_normal, daemon=True).start()

    def _run_normal(self):
        self.log("Extracting normal browsing history...")
        df = extract_browser_history(log_callback=self.log)
        self.display_table(df, title="Browsing History")

    # Security investigation: correlate in-memory alerts with browser history
    def run_security(self):
        threading.Thread(target=self._run_security, daemon=True).start()

    def _run_security(self):
        self.log("Running security investigation (using live alerts)...")
        bd = extract_browser_history(log_callback=self.log)
        with alerts_lock:
            a_copy = alerts_df.copy()
        timeline = correlate_with_alerts(bd, a_copy)
        if timeline.empty:
            self.log("No correlated suspicious activity found.")
            messagebox.showinfo("Security Investigation", "No suspicious activity correlated with alerts.")
        else:
            self.log(f"Found {len(timeline)} correlated events.")
            self.display_table(timeline, title="Suspicious Correlated Events")

    # View raw alerts table
    def view_alerts(self):
        with alerts_lock:
            a_copy = alerts_df.copy()
        if a_copy.empty:
            messagebox.showinfo("Alerts", "No alerts generated yet.")
            return
        self.display_table(a_copy, title="Live Alerts")

    # Generic table display (Treeview)
    def display_table(self, df, title="Data"):
        if df.empty:
            messagebox.showinfo(title, "No data to display.")
            return
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("1100x600")
        columns = df.columns.tolist()
        tree = ttk.Treeview(win, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor='w')
        # Insert rows
        for _, row in df.iterrows():
            values = [row.get(c, "") for c in columns]
            iid = tree.insert("", tk.END, values=values)
            # color tags for certain fields if present
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

# ----------------- Run App -----------------
if __name__ == "__main__":
    # create global reference used by monitor popups
    app = BrowserApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
