import streamlit as st
import numpy as np
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Summarizer", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.block-container { padding-top: 2rem; max-width: 1200px; margin: auto; }
.section-box { padding: 25px; border-radius: 12px; background: #1E1E1E; border: 1px solid #333; margin-bottom: 20px; }
.stButton>button { width: 100%; height: 45px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- VISIBLE TITLE ---
st.title("🤖 AI Text Summarizer")
st.markdown("Summarize long documents using TF-IDF or Smart AI logic.")

# ---------------- SESSION STATE ----------------
if "result" not in st.session_state: st.session_state.result = None
if "text_data" not in st.session_state: st.session_state.text_data = ""
if "page" not in st.session_state: st.session_state.page = "home"

# ---------------- HISTORY FUNCTIONS ----------------
HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                content = f.read()
                return json.loads(content) if content else []
        except:
            return []
    return []

def save_to_history(data):
    history = load_history()
    history.append(data)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

# ---------------- PDF GENERATOR ----------------
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

# ---------------- ALGORITHMS ----------------
def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

def tfidf_summary(sentences, length):
    if len(sentences) <= length: return list(range(len(sentences)))
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    return scores.argsort()[-length:]

def ai_summary(sentences, length):
    if len(sentences) <= length: return list(range(len(sentences)))
    scores = [len(s.split()) + (1/(i+1)) for i, s in enumerate(sentences)]
    return np.array(scores).argsort()[-length:]

# ---------------- NAVIGATION ----------------
nav_col1, nav_col2 = st.columns([1, 1])
with nav_col1:
    if st.button("🏠 Home"):
        st.session_state.page = "home"
        st.rerun()
with nav_col2:
    if st.button("📜 View History"):
        st.session_state.page = "history"
        st.rerun()

# ---------------- PAGE LOGIC ----------------
if st.session_state.page == "history":
    st.header("History")
    history = load_history()

    if not history:
        st.info("Your history is currently empty. Generate a summary to see it here!")
    else:
        for i, item in enumerate(reversed(history), 1):
            with st.expander(f"Summary {i} - {item.get('algorithm', 'Unknown')}"):
                st.write(f"**Length:** {item.get('length', 'N/A')}")
                st.text_area(f"Content {i}", item.get('summary', ""), height=150, disabled=True)
                
                # Dynamic PDF generation for history
                pdf_path = f"history_{i}.pdf"
                save_pdf(item.get('summary', ""), pdf_path)
                with open(pdf_path, "rb") as f:
                    st.download_button(f"Download PDF {i}", f, file_name=f"summary_{i}.pdf", key=f"dl_{i}")
    st.stop()

# ---------------- HOME PAGE (INPUT) ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Input Data")
    uploaded = st.file_uploader("Upload .txt file", type=["txt"])
    
    if uploaded:
        st.session_state.text_data = uploaded.read().decode("utf-8")
    
    text_input = st.text_area("Enter text manually", height=250, value=st.session_state.text_data)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Settings")
    algorithm = st.selectbox("Algorithm", ["TF-IDF", "AI Smart"])
    length = st.number_input("Sentences in Summary", 1, 50, 5)
    
    if st.button("Generate Summary", type="primary"):
        if text_input.strip():
            sentences = split_sentences(text_input)
            with st.spinner("Processing..."):
                idx = tfidf_summary(sentences, length) if algorithm == "TF-IDF" else ai_summary(sentences, length)
                summary_text = "\n".join([sentences[i] for i in sorted(idx)])
                
                new_result = {
                    "algorithm": algorithm,
                    "length": length,
                    "summary": summary_text
                }
                save_to_history(new_result)
                st.session_state.result = new_result
                st.rerun()
        else:
            st.error("Please enter some text first!")

    if st.button("Clear All"):
        st.session_state.result = None
        st.session_state.text_data = ""
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- OUTPUT ----------------
if st.session_state.result:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Result")
    res = st.session_state.result
    st.write(f"**Method:** {res['algorithm']} | **Target Length:** {res['length']} sentences")
    st.success(res["summary"])
    
    pdf = save_pdf(res["summary"], "current_summary.pdf")
    with open(pdf, "rb") as f:
        st.download_button("📥 Download as PDF", f, file_name="summary.pdf")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("Made by Ehsaan Tawakly")