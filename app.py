import streamlit as st
import numpy as np
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="AI Text Summarizer", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.main { padding: 2rem; }
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
}
.stTextArea textarea { border-radius: 10px; }
.section-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #f5f5f5;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "result" not in st.session_state:
    st.session_state.result = None

if "text_data" not in st.session_state:
    st.session_state.text_data = ""

if "page" not in st.session_state:
    st.session_state.page = "home"

# 🔥 IMPORTANT FIX
if "last_algorithm" not in st.session_state:
    st.session_state.last_algorithm = None

# ---------------- HISTORY ----------------
HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_to_history(data):
    history = load_history()
    history.append(data)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

# ---------------- PDF ----------------
def save_pdf(text, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    y = 750
    for line in text.split("\n"):
        if y < 50:
            c.showPage()
            y = 750
        c.drawString(50, y, line[:90])
        y -= 20
    c.save()
    return filename

# ---------------- TEXT PROCESSING ----------------
def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

# ---------------- ALGORITHMS ----------------
def tfidf_summary(sentences, length):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    return sorted(scores.argsort()[-length:])

def ai_summary(sentences, length):
    scores = []
    for i, s in enumerate(sentences):
        score = len(s.split()) + (1 / (i + 1))
        scores.append(score)
    return sorted(np.array(scores).argsort()[-length:])

# ---------------- HISTORY PAGE ----------------
if st.session_state.page == "history":
    st.header("History")

    history = load_history()

    if not history:
        st.warning("No history available")
    else:
        for i, item in enumerate(reversed(history), 1):
            st.markdown(f"### Summary {i}")
            st.write(f"Algorithm: {item['algorithm']}")
            st.write(f"Length: {item['length']}")
            st.text(item["summary"])

            pdf = save_pdf(item["summary"], f"history_{i}.pdf")
            with open(pdf, "rb") as f:
                st.download_button("Download", f, file_name=f"summary_{i}.pdf")

    st.stop()

# ---------------- UI ----------------
st.title("AI Text Summarizer")

col1, col2 = st.columns([2, 1])

# -------- INPUT --------
with col1:
    st.subheader("Input Text")

    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])

    text_input = st.text_area(
        "Paste text",
        height=200,
        value=st.session_state.text_data
    )

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.session_state.text_data = text
    else:
        text = text_input
        st.session_state.text_data = text_input

# -------- SETTINGS --------
with col2:
    st.subheader("Settings")

    algorithm = st.selectbox("Algorithm", ["TF-IDF", "AI Smart Summary"])
    length = st.number_input("Summary Length", 1, 20, 10)

    generate = st.button("Generate")
    clear = st.button("Clear")

# ---------------- CLEAR ----------------
if clear:
    st.session_state.result = None
    st.session_state.text_data = ""
    st.session_state.last_algorithm = None
    st.rerun()

# ---------------- AUTO UPDATE ON ALGO CHANGE 🔥 ----------------
if st.session_state.last_algorithm and st.session_state.last_algorithm != algorithm:

    # Clear previous summary
    st.session_state.result = None

    if text:
        sentences = split_sentences(text)

        if sentences:
            with st.spinner("Generating new summary..."):
                if algorithm == "TF-IDF":
                    idx = tfidf_summary(sentences, length)
                else:
                    idx = ai_summary(sentences, length)

                summary = "\n".join([sentences[i] for i in idx])

            result = {
                "algorithm": algorithm,
                "length": length,
                "summary": summary
            }

            # Save new summary
            st.session_state.result = result
            save_to_history(result)

            st.session_state.last_algorithm = algorithm
            st.rerun()

# update last algorithm
st.session_state.last_algorithm = algorithm

# ---------------- GENERATE BUTTON ----------------
if generate and text:
    sentences = split_sentences(text)

    if sentences:
        with st.spinner("Generating summary..."):
            if algorithm == "TF-IDF":
                idx = tfidf_summary(sentences, length)
            else:
                idx = ai_summary(sentences, length)

            summary = "\n".join([sentences[i] for i in idx])

        result = {
            "algorithm": algorithm,
            "length": length,
            "summary": summary
        }

        st.session_state.result = result
        save_to_history(result)

        st.rerun()

# ---------------- OUTPUT ----------------
if st.session_state.result:
    st.markdown("---")
    st.subheader("Summary Result")

    res = st.session_state.result
    st.write(f"Algorithm: {res['algorithm']}")
    st.write(f"Length: {res['length']}")
    st.text(res["summary"])

    pdf = save_pdf(res["summary"], "summary.pdf")
    with open(pdf, "rb") as f:
        st.download_button("Download PDF", f)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Developed by Ehsaan Tawakly")