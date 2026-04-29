
import re
import time
import requests
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="TruthLens — Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.result-box {
    padding: 1.4rem 1.8rem;
    border-radius: 14px;
    margin: 1rem 0;
    font-size: 1.02rem;
    line-height: 1.7;
    border-left: 6px solid;
}
.real { background:#e6f9ed; border-color:#2ecc71; color:#1a5c33; }
.fake { background:#fdecea; border-color:#e74c3c; color:#7b1c12; }
.sus  { background:#fff8e1; border-color:#f39c12; color:#7d5a00; }

.verdict {
    display:inline-block; padding:6px 20px; border-radius:999px;
    font-weight:700; font-size:1.2rem; margin-bottom:10px;
}
.verdict-real { background:#2ecc71; color:#fff; }
.verdict-fake { background:#e74c3c; color:#fff; }
.verdict-sus  { background:#f39c12; color:#fff; }

.bar-label { font-weight:600; margin-bottom:4px; font-size:0.9rem; }

.source-card {
    background:#f8faff; border:1px solid #e0e7ff; border-radius:10px;
    padding:0.75rem 1rem; margin:0.5rem 0; font-size:0.88rem; line-height:1.5;
}
.source-card a { color:#3730a3; text-decoration:none; font-weight:600; }
.source-card a:hover { text-decoration:underline; }
.source-tag {
    display:inline-block; background:#eef2ff; color:#3730a3;
    border-radius:999px; padding:2px 10px; font-size:0.74rem; margin-bottom:4px;
}
.layer-badge {
    display:inline-block; border-radius:6px; padding:2px 9px;
    font-size:0.74rem; border:1px solid; margin:2px;
}
.layer-ok   { background:#f0fdf4; color:#166534; border-color:#bbf7d0; }
.layer-warn { background:#fefce8; color:#854d0e; border-color:#fde68a; }
.layer-skip { background:#f1f5f9; color:#64748b; border-color:#cbd5e1; }

.stButton > button {
    border-radius:8px !important; font-weight:600 !important;
    transition: all 0.2s ease !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SETTINGS  — replace with your own keys
# --------------------------------------------------
NEWS_API_KEY   = st.secrets.get("e7d75fd233ca4ecab76f2f836b70a807")   # newsapi.org
GNEWS_API_KEY  = st.secrets.get("165fbf819994379eefaede8a541491f4")   # gnews.io

MODEL_PATH = "iamsahhil/fakenews"

HIGH_CONF = 0.72   # above → confident verdict
LOW_CONF  = 0.55   # below → suspicious

CREDIBLE_DOMAINS = {
    "reuters.com", "bbc.com", "bbc.co.uk", "apnews.com",
    "theguardian.com", "nytimes.com", "thehindu.com",
    "ndtv.com", "theprint.in", "aljazeera.com",
    "washingtonpost.com", "ft.com", "economist.com",
}

# --------------------------------------------------
# SESSION STATE — controls text area content
# --------------------------------------------------
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "result"     not in st.session_state:
    st.session_state.result     = None
if "news_data"  not in st.session_state:
    st.session_state.news_data  = None

# --------------------------------------------------
# MODEL LOADER
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model     = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
        truncation=True,
        max_length=512,
    )
    return clf

# --------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------
def classify_news(text: str, clf) -> dict:
    # Run model on full text AND headline (first sentence) for ensemble
    raw_full = clf(text[:512])[0]

    first_sent = re.split(r'(?<=[.!?])\s', text)[0][:512]
    raw_head   = clf(first_sent)[0] if len(first_sent) > 20 else raw_full

    def parse_scores(raw):
        scores    = {r["label"]: r["score"] for r in raw}
        fake_prob = scores.get("FAKE", scores.get("LABEL_0", 0.0))
        real_prob = scores.get("REAL", scores.get("LABEL_1", 0.0))
        total     = fake_prob + real_prob
        if total > 0:
            fake_prob /= total
            real_prob /= total
        return fake_prob, real_prob

    fp_full, rp_full = parse_scores(raw_full)
    fp_head, rp_head = parse_scores(raw_head)

    # Weighted ensemble — full text gets 70%, headline 30%
    fake_prob = 0.70 * fp_full + 0.30 * fp_head
    real_prob = 0.70 * rp_full + 0.30 * rp_head

    # Normalise
    total = fake_prob + real_prob
    if total > 0:
        fake_prob /= total
        real_prob /= total

    winning = max(real_prob, fake_prob)

    if real_prob >= HIGH_CONF:
        label, css = "REAL NEWS",   "real"
    elif fake_prob >= HIGH_CONF:
        label, css = "FAKE NEWS",   "fake"
    elif fake_prob >= LOW_CONF:
        label, css = "SUSPICIOUS",  "sus"
    else:
        label, css = "UNCERTAIN",   "sus"

    return {
        "label":      label,
        "css":        css,
        "real_prob":  real_prob,
        "fake_prob":  fake_prob,
        "confidence": round(winning * 100, 1),
    }

# --------------------------------------------------
# API HELPERS
# --------------------------------------------------
def _build_query(text: str, max_words: int = 7) -> str:
    """Extract a tight keyword query from article text."""
    stops = {"the","a","an","is","are","was","were","in","on","at",
             "of","to","and","or","it","for","with","that","this","has","have"}
    words = [w for w in re.sub(r"[^\w\s]", " ", text).split()
             if w.lower() not in stops and len(w) > 3]
    return " ".join(words[:max_words])


def fetch_newsapi(query: str) -> list:
    """NewsAPI.org — returns list of article dicts."""
    if not NEWS_API_KEY:
        return []
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q":        query,
                "pageSize": 5,
                "language": "en",
                "sortBy":   "relevancy",
                "apiKey":   NEWS_API_KEY,
            },
            timeout=8,
        )
        resp.raise_for_status()
        return resp.json().get("articles", [])
    except Exception as e:
        st.warning(f"NewsAPI error: {e}")
        return []


def fetch_gnews(query: str) -> list:
    """GNews.io — fallback source."""
    if not GNEWS_API_KEY:
        return []
    try:
        resp = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q":        query,
                "max":      5,
                "lang":     "en",
                "token":    GNEWS_API_KEY,
            },
            timeout=8,
        )
        resp.raise_for_status()
        raw = resp.json().get("articles", [])
        # Normalise GNews schema to match NewsAPI schema
        return [
            {
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "url":         a.get("url", ""),
                "source":      {"name": a.get("source", {}).get("name", "Unknown")},
            }
            for a in raw
        ]
    except Exception as e:
        st.warning(f"GNews error: {e}")
        return []


def fetch_news(text: str) -> dict:
    """
    Fetch corroborating articles from NewsAPI (primary) or GNews (fallback).
    Returns dict with articles list + corroboration signal.
    """
    query    = _build_query(text)
    articles = fetch_newsapi(query)

    if not articles:
        articles = fetch_gnews(query)

    if not articles:
        return {"articles": [], "corroborated": None, "credible_count": 0, "query": query}

    credible = [
        a for a in articles
        if any(d in (a.get("url") or "") for d in CREDIBLE_DOMAINS)
    ]

    return {
        "articles":      articles[:5],
        "corroborated":  len(credible) >= 2,
        "credible_count": len(credible),
        "query":         query,
    }


# --------------------------------------------------
# FINAL VERDICT FUSION
# --------------------------------------------------
def fuse_verdict(model_result: dict, news_data: dict) -> dict:
    """
    Combine model probability with news corroboration signal.
    News from credible sources nudges real_prob up; zero credible sources nudges fake_prob up.
    """
    if not news_data or news_data["corroborated"] is None:
        return model_result   # no news signal — trust model as-is

    real_p = model_result["real_prob"]
    fake_p = model_result["fake_prob"]
    cc     = news_data["credible_count"]

    if news_data["corroborated"]:
        # 2+ credible sources → boost real by up to 0.10
        boost  = min(cc * 0.04, 0.10)
        real_p = min(real_p + boost, 0.97)
        fake_p = 1.0 - real_p
    elif cc == 0 and len(news_data["articles"]) > 0:
        # Articles found but none from credible sources → slight fake nudge
        fake_p = min(fake_p + 0.06, 0.97)
        real_p = 1.0 - fake_p

    winning = max(real_p, fake_p)
    if real_p >= HIGH_CONF:
        label, css = "REAL NEWS",  "real"
    elif fake_p >= HIGH_CONF:
        label, css = "FAKE NEWS",  "fake"
    elif fake_p >= LOW_CONF:
        label, css = "SUSPICIOUS", "sus"
    else:
        label, css = "UNCERTAIN",  "sus"

    return {
        "label":      label,
        "css":        css,
        "real_prob":  real_p,
        "fake_prob":  fake_p,
        "confidence": round(winning * 100, 1),
    }


# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
st.title("🔍 TruthLens")
st.caption("AI-powered fake news detector · RoBERTa + News Corroboration")
st.markdown("---")

# Load model
with st.spinner("Loading model…"):
    clf = load_model(MODEL_PATH)
st.success("✅ Model ready")

# --------------------------------------------------
# TEXT INPUT  — bound to session state so Reset clears it
# --------------------------------------------------
article_text = st.text_area(
    "Enter News Headline or Article",
    value=st.session_state.input_text,
    height=220,
    placeholder="Paste a news headline or full article here…",
    key="text_input_widget",
)

col1, col2 = st.columns([3, 1])
with col1:
    analyse_btn = st.button("🔍 Analyse", use_container_width=True,
                             type="primary", disabled=not article_text.strip())
with col2:
    reset_btn = st.button("↺ Reset", use_container_width=True)

# --------------------------------------------------
# RESET — clears state and reruns
# --------------------------------------------------
if reset_btn:
    st.session_state.input_text = ""
    st.session_state.result     = None
    st.session_state.news_data  = None
    st.rerun()

# --------------------------------------------------
# ANALYSE
# --------------------------------------------------
if analyse_btn and article_text.strip():
    st.session_state.input_text = article_text.strip()

    with st.spinner("Analysing article…"):
        model_result = classify_news(article_text.strip(), clf)

    with st.spinner("Checking news sources…"):
        news_data = fetch_news(article_text.strip())

    # Fuse model + news signal
    final_result = fuse_verdict(model_result, news_data)

    st.session_state.result    = final_result
    st.session_state.news_data = news_data

# --------------------------------------------------
# DISPLAY RESULT
# --------------------------------------------------
if st.session_state.result:
    result    = st.session_state.result
    news_data = st.session_state.news_data

    st.markdown("---")
    st.subheader("Analysis Result")

    badge_map = {"real":"verdict-real", "fake":"verdict-fake", "sus":"verdict-sus"}
    badge_cls = badge_map.get(result["css"], "verdict-sus")

    st.markdown(
        f'<span class="verdict {badge_cls}">{result["label"]}</span>',
        unsafe_allow_html=True
    )

    # Confidence box
    st.markdown(
        f'<div class="result-box {result["css"]}">'
        f'<strong>Confidence:</strong> {result["confidence"]}%<br>'
        f'This article has been classified as <strong>{result["label"]}</strong> '
        f'based on language patterns and news corroboration.'
        f'</div>',
        unsafe_allow_html=True
    )

    # Probability bars
    st.markdown("### Model Probability Scores")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="bar-label">✅ REAL</div>', unsafe_allow_html=True)
        st.progress(float(result["real_prob"]))
        st.caption(f'{result["real_prob"]*100:.1f}%')
    with c2:
        st.markdown('<div class="bar-label">❌ FAKE</div>', unsafe_allow_html=True)
        st.progress(float(result["fake_prob"]))
        st.caption(f'{result["fake_prob"]*100:.1f}%')

    # Layer status badges
    st.markdown("### Verification Layers")
    model_badge = '<span class="layer-badge layer-ok">✓ ML Model</span>'
    news_badge  = (
        f'<span class="layer-badge layer-ok">✓ News API ({news_data["credible_count"]} credible)</span>'
        if news_data and news_data["articles"]
        else '<span class="layer-badge layer-skip">– News API (no key / no results)</span>'
    )
    corr_badge = (
        '<span class="layer-badge layer-ok">✓ Corroborated</span>'
        if news_data and news_data.get("corroborated")
        else '<span class="layer-badge layer-warn">⚠ Not corroborated</span>'
        if news_data and news_data["articles"]
        else '<span class="layer-badge layer-skip">– Corroboration skipped</span>'
    )
    st.markdown(model_badge + news_badge + corr_badge, unsafe_allow_html=True)

    # News articles
    if news_data and news_data["articles"]:
        st.markdown("### Related News Articles")
        st.caption(f'Search query used: *"{news_data["query"]}"*')

        for art in news_data["articles"]:
            title  = art.get("title", "No title")
            desc   = art.get("description", "") or ""
            url    = art.get("url", "#")
            source = art.get("source", {}).get("name", "Unknown")
            is_credible = any(d in url for d in CREDIBLE_DOMAINS)
            tag    = "✅ Credible source" if is_credible else "⚠ Unverified source"

            st.markdown(
                f'<div class="source-card">'
                f'<span class="source-tag">{tag} · {source}</span><br>'
                f'<a href="{url}" target="_blank">{title}</a><br>'
                f'<span style="color:#64748b;font-size:0.83rem;">{desc[:160]}{"…" if len(desc)>160 else ""}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("No news articles found. Add your NewsAPI or GNews key in Streamlit secrets to enable corroboration.")

# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------
st.markdown("---")
st.warning(
    "⚠️ This tool is AI-assisted and should not be treated as a final fact-check authority. "
    "Always verify claims with multiple trusted sources."
)
