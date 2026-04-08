import streamlit as st
import numpy as np
import json
import os
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Text Summarizer", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
    }
    .summary-box {
        background-color: #1e2228;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HISTORY ----------------
HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_to_history(summary):
    history = load_history()
    history.append(summary)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

# ---------------- PDF SAVE ----------------
def save_pdf(summary):
    file_path = "summary_output.pdf"
    c = canvas.Canvas(file_path, pagesize=letter)
    y = 750

    for line in summary.split("\n"):
        c.drawString(50, y, line)
        y -= 20

    c.save()
    return file_path

# ---------------- FREQUENCY ----------------
def frequency_scores(sentences):
    word_freq = {}
    for sentence in sentences:
        for word in sentence.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1

    scores = []
    for sentence in sentences:
        score = sum(word_freq[word] for word in sentence.lower().split())
        scores.append(score)

    return np.array(scores)

# ---------------- TITLE ----------------
st.title("🧠 AI Text Summarizer")
st.markdown("✨ Generate smart summaries using TF-IDF & Frequency algorithms")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

summary_length = st.sidebar.slider("Summary Length", 1, 10, 3)
algorithm = st.sidebar.selectbox("Algorithm", ["TF-IDF", "Frequency"])

# ---------------- INPUT SECTION ----------------
st.subheader("📂 Input Text")

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
text_input = st.text_area("Or paste your text here...")

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
elif text_input:
    text = text_input
else:
    text = None

summary_output = ""

# ---------------- PROCESSING ----------------
if text:
    sentences = sent_tokenize(text)

    if len(sentences) > 0:

        with st.spinner("Generating summary... ⏳"):

            if algorithm == "TF-IDF":
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(sentences)
                scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            else:
                scores = frequency_scores(sentences)

            top_indices = scores.argsort()[-summary_length:]
            top_indices = sorted(top_indices)

        st.subheader("📌 Summary Output")

        for i in top_indices:
            st.markdown(f'<div class="summary-box">🔹 {sentences[i]}</div>', unsafe_allow_html=True)
            summary_output += sentences[i] + "\n"

        save_to_history(summary_output)

        # ---------------- ACTION BUTTONS ----------------
        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Download PDF"):
                pdf_file = save_pdf(summary_output)
                with open(pdf_file, "rb") as f:
                    st.download_button("Click to Download", f, file_name="summary.pdf")

        with col2:
            if st.button("📜 View History"):
                history = load_history()
                st.subheader("🕘 Previous Summaries")
                for i, item in enumerate(history, 1):
                    st.markdown(f"**Summary {i}:**")
                    st.write(item)