import streamlit as st
import numpy as np
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="AI Text Summarizer", layout="wide")

# ---------------- CUSTOM CSS (PREMIUM UI) ----------------
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
}
.stTextArea textarea {
    border-radius: 10px;
}
.block-container {
    padding-top: 2rem;
}
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

# ---------------- HEADER ----------------
st.title("AI Text Summarizer")
st.caption("Smart text summarization using NLP")

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
    return sorted(scores.argsort()[-length:])

# ---------------- AI SUMMARY ----------------
def ai_summary(sentences, length):
    scores = []
    for i, s in enumerate(sentences):
        score = len(s.split()) + (1/(i+1))
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

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns([2,1])

# -------- INPUT --------
with col1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Input Text")

    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])

    text_input = st.text_area(
        "Or paste text",
        height=200,
        value=st.session_state.text_data
    )

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.session_state.text_data = text
    else:
        text = text_input
        st.session_state.text_data = text_input

    st.markdown('</div>', unsafe_allow_html=True)

# -------- SETTINGS --------
with col2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Settings")

    algorithm = st.selectbox("Algorithm", ["TF-IDF", "AI Smart Summary"])

    length = st.number_input("Summary Length", 1, 20, 10)

    colA, colB = st.columns(2)

    with colA:
        generate = st.button("Generate")

    with colB:
        clear = st.button("Clear")

    st.markdown('</div>', unsafe_allow_html=True)

# -------- CLEAR --------
if clear:
    st.session_state.result = None
    st.session_state.text_data = ""
    st.rerun()

# -------- GENERATE --------
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

# -------- OUTPUT --------
if st.session_state.result:
    st.markdown("---")
    st.markdown('<div class="section-box">', unsafe_allow_html=True)

    st.subheader("Summary Result")

    res = st.session_state.result
    st.write(f"Algorithm: {res['algorithm']}")
    st.write(f"Length: {res['length']}")
    st.text(res["summary"])

    pdf = save_pdf(res["summary"], "summary.pdf")
    with open(pdf, "rb") as f:
        st.download_button("Download Summary", f)

    st.markdown('</div>', unsafe_allow_html=True)

# -------- FOOTER --------
st.markdown("---")
st.markdown("Developed by Ehsaan Tawakly")