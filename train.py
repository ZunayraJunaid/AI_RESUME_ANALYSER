import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SAMPLE DATASET (replace later with real dataset)
data = {
    "text": [
        "python developer with django flask experience and projects",
        "sales executive with communication skills and marketing",
        "machine learning engineer python ai projects tensorflow",
        "basic resume no skills mentioned",
        "java developer spring boot backend experience",
        "student no experience only education"
    ],
    "label": [2, 1, 2, 0, 2, 0]  # 2=Good, 1=Average, 0=Poor
}

df = pd.DataFrame(data)

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df["text"] = df["text"].apply(clean)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df["text"])

y = df["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model trained and saved!")