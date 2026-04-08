import streamlit as st
import numpy as np
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="AI Text Summarizer", layout="wide")

# ---------------- HEADER ----------------
st.title("AI Text Summarizer")
st.caption("NLP-based text summarization system")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("About")
    st.write("Developed by Ehsaan Tawakly")
    st.write("Supports TF-IDF and Frequency algorithms")

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

# ---------------- FREQUENCY ----------------
def frequency_scores(sentences):
    word_freq = {}
    for sentence in sentences:
        for word in sentence.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1
    return np.array([sum(word_freq[word] for word in s.lower().split()) for s in sentences])

# ---------------- SESSION ----------------
if "results" not in st.session_state:
    st.session_state.results = []

if "show_history" not in st.session_state:
    st.session_state.show_history = False

# ---------------- HISTORY VIEW ----------------
if st.session_state.show_history:
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

    if st.button("Back"):
        st.session_state.show_history = False
    st.stop()

# ---------------- INPUT ----------------
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Input")

    text_input = st.text_area("Enter text", height=200)

with col2:
    st.subheader("Settings")

    algorithm = st.selectbox("Algorithm", ["TF-IDF", "Frequency"])

    length = st.number_input("Summary Length", min_value=1, max_value=20, value=5)

    generate = st.button("Generate Summary")
    view_history = st.button("View History")

if view_history:
    st.session_state.show_history = True
    st.rerun()

# ---------------- GENERATE ----------------
if generate and text_input:

    sentences = split_sentences(text_input)

    if len(sentences) == 0:
        st.warning("No valid sentences found")
    else:
        with st.spinner("Processing..."):

            if algorithm == "TF-IDF":
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(sentences)
                scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            else:
                scores = frequency_scores(sentences)

            top_indices = scores.argsort()[-length:]
            top_indices = sorted(top_indices)

            summary = "\n".join([sentences[i] for i in top_indices])

        # Save result
        result = {
            "algorithm": algorithm,
            "length": length,
            "summary": summary
        }

        st.session_state.results.append(result)
        save_to_history(result)

# ---------------- OUTPUT ----------------
if st.session_state.results:

    st.markdown("---")
    st.header("Generated Summaries")

    for idx, res in enumerate(st.session_state.results, 1):
        st.subheader(f"Summary {idx}")

        st.write(f"Algorithm: {res['algorithm']}")
        st.write(f"Length: {res['length']}")
        st.text(res["summary"])

        pdf_file = save_pdf(res["summary"], f"summary_{idx}.pdf")
        with open(pdf_file, "rb") as f:
            st.download_button("Download", f, file_name=f"summary_{idx}.pdf")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Developed by Ehsaan Tawakly")