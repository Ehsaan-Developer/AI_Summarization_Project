import streamlit as st
import numpy as np
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="AI Text Summarizer", layout="wide")

# ----------- HEADER -----------
st.title("🧠 AI Text Summarizer")
st.caption("Generate smart summaries using NLP techniques")

# ----------- SIDEBAR -----------
with st.sidebar:
    st.title("ℹ️ About")
    st.write("This AI Text Summarizer is developed by **Ehsaan Tawakly**.")
    st.write("It uses NLP techniques like TF-IDF and Frequency scoring.")
    st.markdown("---")
    st.write("⚡ Features:")
    st.write("- TF-IDF Summarization")
    st.write("- Frequency-based Summarization")
    st.write("- PDF Download")
    st.write("- History Tracking")

# ----------- HISTORY -----------
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

# ----------- PDF -----------
def save_pdf(summary):
    file_path = "summary_output.pdf"
    c = canvas.Canvas(file_path, pagesize=letter)
    y = 750

    for line in summary.split("\n"):
        if y < 50:
            c.showPage()
            y = 750
        c.drawString(50, y, line[:90])
        y -= 20

    c.save()
    return file_path

# ----------- SPLIT -----------
def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

# ----------- FREQUENCY -----------
def frequency_scores(sentences):
    word_freq = {}
    for sentence in sentences:
        for word in sentence.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1

    return np.array([
        sum(word_freq[word] for word in s.lower().split())
        for s in sentences
    ])

# ----------- SESSION STATE -----------
if "length" not in st.session_state:
    st.session_state.length = 3

if "algorithm" not in st.session_state:
    st.session_state.algorithm = "TF-IDF"

# ----------- LAYOUT -----------
col1, col2 = st.columns([2, 1])

# -------- INPUT --------
with col1:
    st.subheader("📂 Input")

    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])
    text_input = st.text_area("Or paste text here", height=200)

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
    elif text_input:
        text = text_input
    else:
        text = None

    summary_output = ""

    if text:
        sentences = split_sentences(text)

        if len(sentences) == 0:
            st.warning("⚠️ No valid sentences found.")
        else:
            with st.spinner("Generating summary..."):
                if st.session_state.algorithm == "TF-IDF":
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(sentences)
                    scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
                else:
                    scores = frequency_scores(sentences)

                top_indices = scores.argsort()[-st.session_state.length:]
                top_indices = sorted(top_indices)

            # ✅ AUTO SCROLL TRICK (IMPORTANT)
            st.markdown("### 📌 Summary")

            summary_box = st.container()

            with summary_box:
                for i in top_indices:
                    st.success(sentences[i])
                    summary_output += sentences[i] + "\n"

            save_to_history(summary_output)

            colA, colB = st.columns(2)

            with colA:
                pdf_file = save_pdf(summary_output)
                with open(pdf_file, "rb") as f:
                    st.download_button("💾 Download PDF", f, file_name="summary.pdf")

            with colB:
                if st.button("📜 Show History"):
                    history = load_history()
                    for i, item in enumerate(history, 1):
                        st.info(f"Summary {i}")
                        st.write(item)

# -------- SETTINGS --------
with col2:
    st.subheader("⚙️ Settings")

    st.session_state.length = st.slider("Summary Length", 1, 10, st.session_state.length)

    st.session_state.algorithm = st.radio(
        "Algorithm",
        ["TF-IDF", "Frequency"]
    )

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.write("- Use clean text")
    st.write("- Large text = better summary")

# -------- FOOTER --------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Developed by <b>Ehsaan Tawakly</b> 🚀"
    "</div>",
    unsafe_allow_html=True
)