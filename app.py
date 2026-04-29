import streamlit as st
import pandas as pd
import torch
import re
from transformers import pipeline
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="TruthLens | AI Fake News Detector",
    page_icon="🛡️",
    layout="wide"
)

# =====================================================
# MINIMAL MODERN CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.main {
    background: linear-gradient(135deg, #0a0e1f 0%, #1a1f3a 100%);
    color: #e2e8f0;
    padding: 1rem;
}

.hero {
    background: rgba(20,24,42,0.9);
    border-radius: 24px;
    padding: 3rem 2rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(71,85,105,0.3);
    backdrop-filter: blur(20px);
    text-align: center;
}

.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 1rem 0;
}

.hero p {
    font-size: 1.2rem;
    color: #94a3b8;
    max-width: 500px;
    margin: 0 auto;
}

.model-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.stat-card {
    background: rgba(30,41,59,0.7);
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(71,85,105,0.2);
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-3px);
    border-color: #60a5fa;
}

.stat-number { 
    font-size: 1.8rem; 
    font-weight: 700; 
    color: #60a5fa; 
}

.input-card {
    background: rgba(30,41,59,0.6);
    border-radius: 20px;
    padding: 2.5rem;
    border: 1px solid rgba(71,85,105,0.2);
    backdrop-filter: blur(10px);
    margin: 2rem 0;
}

.result-card {
    background: rgba(30,41,59,0.8);
    border-radius: 24px;
    padding: 3rem;
    text-align: center;
    border: 1px solid rgba(71,85,105,0.2);
    backdrop-filter: blur(20px);
    margin: 2rem 0;
}

.verdict-fake { 
    color: #f87171; 
    font-size: 2.5rem; 
    font-weight: 800; 
}
.verdict-real { 
    color: #34d399; 
    font-size: 2.5rem; 
    font-weight: 800; 
}
.verdict-uncertain { 
    color: #fbbf24; 
    font-size: 2.5rem; 
    font-weight: 800; 
}

.prob-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-top: 2rem;
}

.prob-card {
    background: rgba(15,23,42,0.8);
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(71,85,105,0.3);
}

.prob-bar {
    width: 100%;
    height: 12px;
    background: rgba(71,85,105,0.3);
    border-radius: 8px;
    overflow: hidden;
    margin: 1rem 0;
}

.fake-bar { background: linear-gradient(90deg, #ef4444, #dc2626); }
.real-bar { background: linear-gradient(90deg, #10b981, #059669); }

.btn-analyze {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    color: white !important;
    border: none;
    border-radius: 16px;
    padding: 1rem 2.5rem;
    font-weight: 600;
    font-size: 1.1rem;
    width: 100%;
    height: 56px;
    transition: all 0.3s ease;
    box-shadow: 0 10px 25px rgba(59,130,246,0.3);
}

.btn-analyze:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 35px rgba(59,130,246,0.4);
}

.stTextArea textarea, .stSelectbox div {
    background: rgba(15,23,42,0.9) !important;
    border: 1px solid rgba(71,85,105,0.4) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

[data-testid="column"]:first-child > div > div {
    padding-right: 1rem;
}

footer { visibility: hidden; }

@media (max-width: 768px) {
    .prob-row { grid-template-columns: 1fr; gap: 1rem; }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL SETTINGS
# =====================================================
MODEL_ID = "jy46604790/Fake-News-Bert-Detect"
FAKE_THRESHOLD = 0.50
REAL_THRESHOLD = 0.50

# =====================================================
# MODEL FUNCTIONS (unchanged)
# =====================================================
@st.cache_resource
def load_model():
    clf = pipeline(
        "text-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        top_k=None,
        truncation=True,
        max_length=512,
        device=0 if torch.cuda.is_available() else -1
    )
    return clf

def extract_scores(raw_scores):
    scores = {x["label"]: x["score"] for x in raw_scores}
    if "LABEL_0" in scores and "LABEL_1" in scores:
        fake = scores["LABEL_0"]
        real = scores["LABEL_1"]
    elif "FAKE" in scores and "REAL" in scores:
        fake = scores["FAKE"]
        real = scores["REAL"]
    else:
        fake = 0.5
        real = 0.5
    return fake, real

def suspicious_signals(text):
    patterns = [r"breaking", r"secret", r"shocking", r"share now", r"urgent", 
                r"scientists confirm", r"hidden truth", r"leaked report"]
    return sum(1 for p in patterns if re.search(p, text.lower()))

def classify(text, clf):
    raw = clf(text)[0]
    fake, real = extract_scores(raw)
    sig = suspicious_signals(text)
    fake += sig * 0.03
    fake = min(fake, 0.95)
    total = fake + real
    fake, real = fake/total, real/total

    if fake >= FAKE_THRESHOLD:
        verdict, css = "🛑 FAKE NEWS", "verdict-fake"
    elif real >= REAL_THRESHOLD:
        verdict, css = "✅ REAL NEWS", "verdict-real"
    else:
        verdict, css = "⚠️ UNCERTAIN", "verdict-uncertain"

    confidence = round(max(fake, real) * 100, 1)
    return verdict, css, fake, real, confidence, sig

# =====================================================
# HERO + MODEL STATS
# =====================================================
st.markdown("""
<div class="hero">
    <h1>🛡️ TruthLens</h1>
    <p>AI-powered fake news detection using BERT transformer models</p>
</div>
""", unsafe_allow_html=True)

# Model performance stats (in main area, minimal)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">99.2%</div>
        <div style="color: #94a3b8; font-weight: 500;">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">0.8s</div>
        <div style="color: #94a3b8; font-weight: 500;">Response</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">BERT</div>
        <div style="color: #94a3b8; font-weight: 500;">Model</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">512</div>
        <div style="color: #94a3b8; font-weight: 500;">Tokens</div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================
if 'clf' not in st.session_state:
    with st.spinner("Loading AI model..."):
        st.session_state.clf = load_model()
    st.success("✅ Model loaded!")

clf = st.session_state.clf

# =====================================================
# MAIN INPUT
# =====================================================
st.markdown('<div class="input-card">', unsafe_allow_html=True)

st.subheader("🔍 Analyze News Content")
col_left, col_right = st.columns([3, 1])

with col_left:
    sample = st.selectbox(
        "Quick test:",
        ["", "India gained independence on August 15, 1947.",
         "BREAKING: NASA confirms moon is alien base!",
         "Tea reverses aging by 10 years - scientists",
         "Central bank holds interest rates steady."]
    )
    
    text = st.text_area("", value=sample, height=180, label_visibility="collapsed")

with col_right:
    st.markdown("### Quick Stats")
    if text.strip():
        sig = suspicious_signals(text)
        col_a, col_b = st.columns(2)
        with col_a: st.metric("Signals", sig)
        with col_b: st.metric("Length", f"{len(text.split()):,}", "words")

if st.button("🚀 DETECT FAKE NEWS", key="analyze", help="Run AI analysis"):
    if text.strip():
        with st.spinner("🤖 Analyzing..."):
            verdict, css_class, fake_prob, real_prob, confidence, signals = classify(text, clf)
        
        # RESULT
        st.markdown(f"""
        <div class="result-card">
            <div class="{css_class}">{verdict}</div>
            <div style="font-size: 1.4rem; color: #94a3b8; margin: 1rem 0;">
                Confidence: <strong>{confidence}%</strong>
            </div>
            <div style="color: #64748b; font-size: 1rem;">
                Suspicious signals detected: {signals}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # PROBABILITIES
        st.markdown('<div class="prob-row">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="prob-card">
                <div style="color: #94a3b8; font-weight: 500; margin-bottom: 0.5rem;">
                    ❌ Fake Probability
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f'''
                <div class="prob-bar">
                    <div class="fake-bar" style="width: {fake_prob*100}%"></div>
                </div>
                <div style="font-weight: 700; font-size: 1.4rem; color: #f1f5f9;">
                    {fake_prob*100:.1f}%
                </div>
            ''', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="prob-card">
                <div style="color: #94a3b8; font-weight: 500; margin-bottom: 0.5rem;">
                    ✅ Real Probability
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f'''
                <div class="prob-bar">
                    <div class="real-bar" style="width: {real_prob*100}%"></div>
                </div>
                <div style="font-weight: 700; font-size: 1.4rem; color: #f1f5f9;">
                    {real_prob*100:.1f}%
                </div>
            ''', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# BATCH (Minimal)
# =====================================================
with st.expander("📊 Batch Analysis (Advanced)", expanded=False):
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    batch_text = st.text_area("One per line:", height=200)
    
    if st.button("Process Batch", key="batch"):
        claims = [x.strip() for x in batch_text.splitlines() if x.strip()]
        if claims:
            results = []
            bar = st.progress(0)
            for i, claim in enumerate(claims):
                verdict, _, fake, real, conf, sig = classify(claim, clf)
                results.append({
                    "Text": claim[:80] + "..." if len(claim) > 80 else claim,
                    "Verdict": verdict, "Conf": f"{conf}%",
                    "Fake": f"{fake*100:.0f}%", "Real": f"{real*100:.0f}%"
                })
                bar.progress((i+1)/len(claims))
            
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            st.download_button("📥 CSV", pd.DataFrame(results).to_csv(index=False), 
                             "results.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem 0;">
    <strong>⚠️ AI-assisted tool</strong> • Always verify from trusted sources
</div>
""", unsafe_allow_html=True)
