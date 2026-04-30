import streamlit as st
import pandas as pd
import torch
import re
from transformers import pipeline


st.set_page_config(
    page_title="Cloud-Based Fake News Detection Using NLP",
    page_icon="🔍",
    layout="centered"
)

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)


MODEL_ID = "iamsahhil/Fake-News-Bert-Detect"

FAKE_THRESHOLD = 0.50
REAL_THRESHOLD = 0.50


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


def fake_signals(text):
    patterns = [
        r"breaking",
        r"secret",
        r"shocking",
        r"share now",
        r"hidden truth",
        r"scientists confirm",
        r"they don't want you to know",
        r"urgent",
        r"leaked report"
    ]

    count = 0
    for p in patterns:
        if re.search(p, text.lower()):
            count += 1

    return count


def classify(text, clf):
    raw = clf(text)[0]

    fake_prob, real_prob = extract_fake_real(raw)

   
    signals = fake_signals(text)
    fake_prob += signals * 0.04
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



st.title("Cloud-Based Fake News Detection Using NLP")


with st.spinner("Loading AI model..."):
    clf = load_model()

st.success("Model Loaded Successfully")

st.markdown("---")

tabs = st.tabs(["🔍 Single Check", "📋 Batch Check"])


with tabs[0]:

    example = st.selectbox(
        "Try Example",
        [
            "",
            "NASA confirms moon is alien spaceship hidden for centuries",
            "India gained independence on August 15, 1947",
            "Scientists say tea reverses aging by 10 years",
            "Federal Reserve kept rates unchanged",
            "5G towers secretly spread viruses share now"
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

            st.caption(f"Suspicious signals found: {result['signals']}")

# =====================================================
# BATCH
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
