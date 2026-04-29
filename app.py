import streamlit as st
import pandas as pd
import torch
import re
from transformers import pipeline

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="TruthLens | Cloud-Based Fake News Detection Using NLP",
    page_icon="🛡️",
    layout="wide"
)

# =====================================================
# MODERN CSS UI
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.main {
    background: linear-gradient(135deg,#0f172a,#111827);
    color:white;
}

.hero-box {
    padding:30px;
    border-radius:18px;
    background: linear-gradient(135deg,#1e293b,#0f172a);
    color:white;
    text-align:center;
    margin-bottom:25px;
    box-shadow:0 10px 25px rgba(0,0,0,0.25);
}

.hero-title{
    font-size:38px;
    font-weight:700;
    margin-bottom:5px;
}

.hero-sub{
    font-size:17px;
    color:#cbd5e1;
}

.result-box{
    padding:22px;
    border-radius:16px;
    margin-top:18px;
    text-align:center;
    font-size:20px;
    font-weight:600;
}

.fake{
    background:#7f1d1d;
    color:white;
}

.real{
    background:#14532d;
    color:white;
}

.uncertain{
    background:#78350f;
    color:white;
}

.metric-card{
    padding:15px;
    border-radius:14px;
    background:#111827;
    color:white;
    text-align:center;
}

.stButton>button{
    background:linear-gradient(90deg,#2563eb,#06b6d4);
    color:white;
    border:none;
    border-radius:12px;
    padding:12px;
    font-weight:600;
}

.stTextArea textarea{
    border-radius:12px;
}

footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL SETTINGS
# =====================================================
MODEL_ID = "jy46604790/Fake-News-Bert-Detect"

FAKE_THRESHOLD = 0.50
REAL_THRESHOLD = 0.50

# =====================================================
# LOAD MODEL
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

# =====================================================
# EXTRACT LABELS
# =====================================================
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

# =====================================================
# EXTRA SIGNALS
# =====================================================
def suspicious_signals(text):
    patterns = [
        r"breaking",
        r"secret",
        r"shocking",
        r"share now",
        r"urgent",
        r"scientists confirm",
        r"hidden truth",
        r"leaked report"
    ]

    count = 0
    for p in patterns:
        if re.search(p, text.lower()):
            count += 1
    return count

# =====================================================
# CLASSIFIER
# =====================================================
def classify(text, clf):
    raw = clf(text)[0]

    fake, real = extract_scores(raw)

    sig = suspicious_signals(text)

    fake += sig * 0.03
    fake = min(fake, 0.95)

    total = fake + real
    fake /= total
    real /= total

    if fake >= FAKE_THRESHOLD:
        verdict = "FAKE NEWS"
        css = "fake"

    elif real >= REAL_THRESHOLD:
        verdict = "REAL NEWS"
        css = "real"

    else:
        verdict = "UNCERTAIN"
        css = "uncertain"

    confidence = round(max(fake, real) * 100, 1)

    return verdict, css, fake, real, confidence, sig

# =====================================================
# HERO SECTION
# =====================================================
st.markdown("""
<div class="hero-box">
<div class="hero-title">🛡️ TruthLens</div>
<div class="hero-sub">
Cloud-Based Fake News Detection Using Natural Language Processing
</div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================
with st.spinner("Loading NLP model..."):
    clf = load_model()

st.success("AI Model Loaded Successfully")

# =====================================================
# TABS
# =====================================================
tab1, tab2 = st.tabs(["🔍 Detect News", "📋 Batch Analysis"])

# =====================================================
# SINGLE DETECTION
# =====================================================
with tab1:

    st.subheader("Check Headline or Article")

    sample = st.selectbox(
        "Try Sample Input",
        [
            "",
            "India gained independence on August 15, 1947.",
            "NASA confirms moon is alien spaceship.",
            "Scientists say tea reverses aging by 10 years.",
            "The central bank kept interest rates unchanged."
        ]
    )

    text = st.text_area(
        "Paste News Content",
        value=sample,
        height=220
    )

    if st.button("🔍 Analyse News", use_container_width=True):

        if text.strip():

            verdict, css, fake, real, conf, sig = classify(text, clf)

            st.markdown(
                f"""
                <div class="result-box {css}">
                {verdict}<br>
                Confidence: {conf}%
                </div>
                """,
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)

            with c1:
                st.markdown('<div class="metric-card">❌ Fake Probability</div>', unsafe_allow_html=True)
                st.progress(float(fake))
                st.write(f"{fake*100:.1f}%")

            with c2:
                st.markdown('<div class="metric-card">✅ Real Probability</div>', unsafe_allow_html=True)
                st.progress(float(real))
                st.write(f"{real*100:.1f}%")

            st.info(f"Suspicious Signals Detected: {sig}")

# =====================================================
# BATCH
# =====================================================
with tab2:

    st.subheader("Multiple Claim Detection")

    batch = st.text_area(
        "Enter one news item per line",
        height=250
    )

    if st.button("📊 Run Batch Analysis", use_container_width=True):

        claims = [x.strip() for x in batch.splitlines() if x.strip()]

        rows = []

        for claim in claims:
            verdict, css, fake, real, conf, sig = classify(claim, clf)

            rows.append({
                "Claim": claim,
                "Verdict": verdict,
                "Confidence %": conf,
                "Fake %": round(fake*100,1),
                "Real %": round(real*100,1)
            })

        df = pd.DataFrame(rows)

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇ Download Results CSV",
            df.to_csv(index=False),
            file_name="truthlens_results.csv",
            mime="text/csv"
        )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.warning("⚠️ This system is AI-assisted. Always verify important claims from trusted official sources.")
