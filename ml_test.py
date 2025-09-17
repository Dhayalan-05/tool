import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import scipy

# Load CSV
df = pd.read_csv("browsing_history.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Convert timestamp to datetime and extract hour
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour

# Ensure duration is numeric
df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

# Features and target
X = df[["url", "hour", "duration"]]
y = df["category"]

# Vectorize URL
tfidf = TfidfVectorizer()
X_url = tfidf.fit_transform(X["url"])

# Combine with numeric columns
X_numeric = X[["hour", "duration"]].values
X_final = scipy.sparse.hstack([X_url, X_numeric])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
