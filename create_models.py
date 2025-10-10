import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# Example data (replace with your real data)
corpus = [
    "This is an educational website",
    "Shopping is fun",
    "Social media is addictive",
    "Learn Python programming",
    "Buy gadgets online"
]

labels = ["Education", "Shopping", "Social", "Education", "Shopping"]
# -------------------------

# Create and train TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(corpus)

# Save TF-IDF Vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X, labels)

# Save Random Forest model
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("Files 'rf_model.pkl' and 'tfidf_vectorizer.pkl' have been created!")
