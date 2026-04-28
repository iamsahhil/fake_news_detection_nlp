import re
import requests
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; }

.result-box {
    padding: 1.4rem 1.8rem;
    border-radius: 14px;
    margin: 1rem 0;
    font-size: 1.05rem;
    line-height: 1.7;
    border-left: 6px solid;
}

.real  { background:#e6f9ed; border-color:#2ecc71; color:#1a5c33; }
.fake  { background:#fdecea; border-color:#e74c3c; color:#7b1c12; }
.sus   { background:#fff8e1; border-color:#f39c12; color:#7d5a00; }

.verdict {
    display:inline-block;
    padding:6px 18px;
    border-radius:999px;
    font-weight:700;
    font-size:1.2rem;
    margin-bottom:10px;
}

.verdict-real { background:#2ecc71; color:#fff; }
.verdict-fake { background:#e74c3c; color:#fff; }
.verdict-sus  { background:#f39c12; color:#fff; }

.bar-label { font-weight:600; margin-bottom:4px; }

.source-pill {
    display:inline-block;
    background:#eef2ff;
    color:#3730a3;
    border-radius:999px;
    padding:3px 12px;
    font-size:0.78rem;
    margin:3px 3px 3px 0;
}

.api-badge {
    display:inline-block;
    background:#f0fdf4;
    color:#166534;
    border-radius:6px;
    padding:2px 9px;
    font-size:0.76rem;
    border:1px solid #bbf7d0;
    margin-left:6px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
news_api_key = "e7d75fd233ca4ecab76f2f836b70a807"
gnews_api_key = "165fbf819994379eefaede8a541491f4"

model_path = "iamsahhil/fakenews"

high_conf = 0.75
low_conf = 0.55

# --------------------------------------------------
# MODEL LOADER
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path):

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)

    model.eval()

    device = 0 if torch.cuda.is_available() else -1

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
        truncation=True,
        max_length=512
    )

    return clf

# --------------------------------------------------
# CLASSIFICATION LOGIC
# --------------------------------------------------
def classify_news(text, clf):

    raw = clf(text)[0]
    scores = {r["label"]: r["score"] for r in raw}

    fake_prob = scores.get("FAKE", scores.get("LABEL_0", 0.0))
    real_prob = scores.get("REAL", scores.get("LABEL_1", 0.0))

    total = fake_prob + real_prob

    if total > 0:
        fake_prob /= total
        real_prob /= total

    gap = abs(real_prob - fake_prob)
    winning_prob = max(real_prob, fake_prob)

    if real_prob >= 0.60:
        label = "REAL NEWS"
        css = "real"

    elif fake_prob >= 0.60:
        label = "FAKE NEWS"
        css = "fake"

    else:
        label = "SUSPICIOUS"
        css = "sus"

    return {
        "label": label,
        "css": css,
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "gap": gap,
        "confidence": round(winning_prob * 100, 1)
    }

# --------------------------------------------------
# API HELPERS
# --------------------------------------------------
def fetch_newsapi(query):

    if news_api_key == "YOUR_NEWS_API_KEY":
        return []

    try:
        url = "https://newsapi.org/v2/everything"

        params = {
            "q": query,
            "pageSize": 4,
            "language": "en",
            "apiKey": news_api_key
        }

        r = requests.get(url, params=params, timeout=8)
        data = r.json()

        return data.get("articles", [])

    except:
        return []

# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
st.title("Fake News Detector")
st.caption("Powered by fine-tuned RoBERTa")

st.markdown("---")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
with st.spinner("Loading model..."):
    clf = load_model(model_path)

st.success("Model loaded successfully!")

# --------------------------------------------------
# INPUT
# --------------------------------------------------
article_text = st.text_area(
    "Enter News Headline / Article",
    height=220,
    placeholder="Paste article or headline here..."
)

# --------------------------------------------------
# BUTTONS
# --------------------------------------------------
col1, col2 = st.columns([2,1])

with col1:
    analyse_btn = st.button("Analyse Article", use_container_width=True)

with col2:
    reset_btn = st.button("Reset", use_container_width=True)

# --------------------------------------------------
# RESET
# --------------------------------------------------
if reset_btn:
    st.rerun()

# --------------------------------------------------
# ANALYZE
# --------------------------------------------------
if analyse_btn:

    text = article_text.strip()

    if not text:
        st.warning("Please enter article text.")

    else:

        with st.spinner("Analysing..."):

            result = classify_news(text, clf)

        st.markdown("---")
        st.subheader("Analysis Result")

        badge_cls = {
            "real": "verdict-real",
            "fake": "verdict-fake",
            "sus": "verdict-sus"
        }[result["css"]]

        st.markdown(
            f'<span class="verdict {badge_cls}">{result["label"]}</span>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-box {result["css"]}">'
            f'Confidence: {result["confidence"]}%'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("### Model Probability Scores")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="bar-label">REAL</div>', unsafe_allow_html=True)
            st.progress(result["real_prob"])
            st.caption(f'{result["real_prob"]*100:.1f}%')

        with c2:
            st.markdown('<div class="bar-label">FAKE</div>', unsafe_allow_html=True)
            st.progress(result["fake_prob"])
            st.caption(f'{result["fake_prob"]*100:.1f}%')

# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------
st.markdown("---")
st.warning(
    "This tool is AI-assisted and should not be considered a final fact-check authority."
)
