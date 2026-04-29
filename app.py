import streamlit as st
import pandas as pd
import torch
import re
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="TruthLens v2",
    page_icon="🔍",
    layout="centered"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
body {
    font-family: Inter, sans-serif;
}
.big-box {
    padding: 20px;
    border-radius: 14px;
    margin-top: 10px;
    margin-bottom: 15px;
}
.fake-box {
    background: #fff1f2;
    border-left: 6px solid #ef4444;
}
.real-box {
    background: #ecfdf5;
    border-left: 6px solid #22c55e;
}
.uncertain-box {
    background: #fffbeb;
    border-left: 6px solid #f59e0b;
}
.small {
    font-size: 14px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SETTINGS
# =====================================================
MODEL_ID = "iamsahhil/fakenews"

FAKE_THRESHOLD = 0.42
REAL_THRESHOLD = 0.58

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True,
        max_length=512,
        device=0 if torch.cuda.is_available() else -1
    )

    return clf

# =====================================================
# LABEL FIXER
# =====================================================
def extract_fake_real(raw_scores):
    scores = {x["label"]: x["score"] for x in raw_scores}

    if "LABEL_0" in scores and "LABEL_1" in scores:
        fake_prob = scores["LABEL_0"]
        real_prob = scores["LABEL_1"]

    elif "FAKE" in scores and "REAL" in scores:
        fake_prob = scores["FAKE"]
        real_prob = scores["REAL"]

    else:
        fake_prob = 0.5
        real_prob = 0.5

    return fake_prob, real_prob

# =====================================================
# OPTIONAL SIGNAL CHECKS
# =====================================================
def fake_signals(text):
    patterns = [
        r"breaking",
        r"secret",
        r"shocking",
        r"share now",
        r"hidden truth",
        r"scientists confirm",
        r"they don't want you to know"
    ]

    count = 0
    for p in patterns:
        if re.search(p, text.lower()):
            count += 1

    return count

# =====================================================
# MAIN CLASSIFIER
# =====================================================
def classify(text, clf):
    raw = clf(text)[0]

    fake_prob, real_prob = extract_fake_real(raw)

    # small fake boost if suspicious words found
    signals = fake_signals(text)

    fake_prob += signals * 0.03
    fake_prob = min(fake_prob, 0.95)

    total = fake_prob + real_prob
    fake_prob /= total
    real_prob /= total

    if fake_prob >= FAKE_THRESHOLD:
        verdict = "FAKE NEWS"
        css = "fake"

    elif real_prob >= REAL_THRESHOLD:
        verdict = "REAL NEWS"
        css = "real"

    else:
        verdict = "UNCERTAIN"
        css = "uncertain"

    confidence = round(max(fake_prob, real_prob) * 100, 1)

    return {
        "verdict": verdict,
        "css": css,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "confidence": confidence,
        "signals": signals
    }

# =====================================================
# UI
# =====================================================
st.title("🔍 TruthLens v2")
st.caption("AI Fake News Detector using RoBERTa")

with st.spinner("Loading AI model..."):
    clf = load_model()

st.success("Model Loaded Successfully")

st.markdown("---")

tabs = st.tabs(["🔍 Single Check", "📋 Batch Check"])

# =====================================================
# SINGLE CHECK
# =====================================================
with tabs[0]:

    example = st.selectbox(
        "Try Example",
        [
            "",
            "NASA confirms moon is alien spaceship hidden for centuries",
            "India gained independence on August 15, 1947",
            "Scientists say tea reverses aging by 10 years",
            "The Federal Reserve kept interest rates unchanged"
        ]
    )

    text = st.text_area(
        "Paste headline or article",
        value=example,
        height=180
    )

    if st.button("Analyse", use_container_width=True):

        if text.strip():

            result = classify(text, clf)

            if result["css"] == "fake":
                box = "fake-box"
                emoji = "❌"

            elif result["css"] == "real":
                box = "real-box"
                emoji = "✅"

            else:
                box = "uncertain-box"
                emoji = "⚠️"

            st.markdown(
                f"""
                <div class="big-box {box}">
                <h2>{emoji} {result["verdict"]}</h2>
                <p><b>Confidence:</b> {result["confidence"]}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("Probability Scores")

            c1, c2 = st.columns(2)

            with c1:
                st.write("❌ Fake")
                st.progress(float(result["fake_prob"]))
                st.write(f"{result['fake_prob']*100:.1f}%")

            with c2:
                st.write("✅ Real")
                st.progress(float(result["real_prob"]))
                st.write(f"{result['real_prob']*100:.1f}%")

            st.caption(f"Suspicious signals detected: {result['signals']}")

# =====================================================
# BATCH CHECK
# =====================================================
with tabs[1]:

    batch = st.text_area(
        "Enter one claim per line",
        height=220
    )

    if st.button("Run Batch", use_container_width=True):

        rows = []

        claims = [x.strip() for x in batch.splitlines() if x.strip()]

        for claim in claims:
            r = classify(claim, clf)

            rows.append({
                "Claim": claim,
                "Verdict": r["verdict"],
                "Confidence": r["confidence"],
                "Fake %": round(r["fake_prob"] * 100, 1),
                "Real %": round(r["real_prob"] * 100, 1)
            })

        df = pd.DataFrame(rows)

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            file_name="results.csv",
            mime="text/csv"
        )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.warning("Always verify important claims using trusted sources.")
