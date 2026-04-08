import streamlit as st
import numpy as np
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="AI Text Summarizer", layout="wide")

# ---------------- SESSION ----------------
if "result" not in st.session_state:
    st.session_state.result = None

if "page" not in st.session_state:
    st.session_state.page = "home"

if "text_data" not in st.session_state:
    st.session_state.text_data = ""

# ---------------- HEADER ----------------
st.title("AI Text Summarizer")
st.caption("TF-IDF and AI-based Smart Summarization")

# ---------------- NAVIGATION ----------------
col_nav1, col_nav2 = st.columns([1,1])

with col_nav1:
    if st.button("Home"):
        st.session_state.page = "home"
        st.rerun()

with col_nav2:
    if st.button("History"):
        st.session_state.page = "history"
        st.rerun()

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

# ---------------- SPLIT ----------------
def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

# ---------------- TF-IDF ----------------
def tfidf_summary(sentences, length):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    top_indices = scores.argsort()[-length:]
    return sorted(top_indices)

# ---------------- AI SMART SUMMARY ----------------
def ai_smart_summary(sentences, length):
    # improved scoring using sentence length + position + frequency
    scores = []
    for i, sentence in enumerate(sentences):
        length_score = len(sentence.split())
        position_score = 1 / (i + 1)
        score = length_score + position_score
        scores.append(score)

    scores = np.array(scores)
    top_indices = scores.argsort()[-length:]
    return sorted(top_indices)

# ---------------- HISTORY PAGE ----------------
if st.session_state.page == "history":
    st.header("History")

    history = load_history()

    if not history:
        st.warning("No history available")
    else:
        for i, item in enumerate(reversed(history), 1):
            st.subheader(f"Summary {i}")
            st.write(f"Algorithm: {item['algorithm']}")
            st.write(f"Length: {item['length']}")
            st.text(item["summary"])

            pdf = save_pdf(item["summary"], f"history_{i}.pdf")
            with open(pdf, "rb") as f:
                st.download_button("Download", f, file_name=f"summary_{i}.pdf")

    st.stop()

# ---------------- HOME PAGE ----------------
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Input")

    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])

    text_input = st.text_area(
        "Or enter text manually",
        height=200,
        value=st.session_state.text_data
    )

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.session_state.text_data = text
    else:
        text = text_input
        st.session_state.text_data = text_input

with col2:
    st.subheader("Settings")

    algorithm = st.selectbox("Algorithm", ["TF-IDF", "AI Smart Summary"])

    length = st.number_input(
        "Summary Length",
        min_value=1,
        max_value=20,
        value=10
    )

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        generate = st.button("Generate Summary")

    with col_btn2:
        clear = st.button("Clear")

# ---------------- CLEAR ----------------
if clear:
    st.session_state.result = None
    st.session_state.text_data = ""
    st.rerun()

# ---------------- GENERATE ----------------
if generate and text:

    sentences = split_sentences(text)

    if len(sentences) == 0:
        st.warning("No valid sentences found")
    else:
        with st.spinner("Processing..."):

            if algorithm == "TF-IDF":
                indices = tfidf_summary(sentences, length)
            else:
                indices = ai_smart_summary(sentences, length)

            summary = "\n".join([sentences[i] for i in indices])

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
    st.header("Generated Summary")

    res = st.session_state.result

    st.write(f"Algorithm: {res['algorithm']}")
    st.write(f"Length: {res['length']}")
    st.text(res["summary"])

    pdf_file = save_pdf(res["summary"], "summary.pdf")
    with open(pdf_file, "rb") as f:
        st.download_button("Download", f, file_name="summary.pdf")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Developed by Ehsaan Tawakly")