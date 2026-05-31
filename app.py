from flask import Flask, render_template, request
from pdfminer.high_level import extract_text
import os
import pickle
import re
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"

os.makedirs("uploads", exist_ok=True)

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))


# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template("index.html")


# ---------------- ANALYZE ----------------
@app.route('/analyze', methods=['POST'])
def analyze():

    file = request.files.get("resume")

    if not file or file.filename == "":
        return "No file uploaded"

    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    text = extract_text(path)

    cleaned = clean_text(text)

    X = vectorizer.transform([cleaned])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X).max()

    score = int(prob * 100)

    if pred == 2:
        status = "🔥 Strong Resume"
        feedback = ["Good technical + project profile"]
    elif pred == 1:
        status = "⚠️ Average Resume"
        feedback = ["Improve skills and projects"]
    else:
        status = "❌ Weak Resume"
        feedback = ["Add skills, experience, projects"]

    return render_template("result.html",
                           score=score,
                           status=status,
                           feedback=feedback)


# ---------------- CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)