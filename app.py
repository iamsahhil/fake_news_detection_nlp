"""
Fake News Detection App
Built with RoBERTa + Streamlit
Final Year Academic Project
"""

import os
import re
import math
import time
import requests
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* General */
body { font-family: 'Segoe UI', sans-serif; }

/* Result cards */
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

/* Confidence bar label */
.bar-label { font-weight:600; margin-bottom:4px; }

/* Verdict badge */
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

/* Source pill */
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


# ──────────────────────────────────────────────
# CONFIGURATION SETTINGS
# ──────────────────────────────────────────────
# Previously in sidebar, now set directly
news_api_key = ("e7d75fd233ca4ecab76f2f836b70a807")
gnews_api_key = ("165fbf819994379eefaede8a541491f4")
model_path = "iamsahhil/fakenews"
high_conf = 0.75
low_conf = 0.55
# ──────────────────────────────────────────────
# MODEL LOADER (cached)
# ──────────────────────────────────────────────
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
        top_k=None,          # return ALL label scores
        truncation=True,
        max_length=512,
    )
    return clf


# ──────────────────────────────────────────────
# IMPROVED PREDICTION LOGIC
# ──────────────────────────────────────────────
def classify_news(text: str, clf, high_thresh: float, low_thresh: float) -> dict:
    """
    Run the RoBERTa classifier and apply a calibrated decision rule.

    Decision rule
    ─────────────
    The raw softmax outputs are not always well-calibrated, so we apply
    a confidence-gap check instead of a single threshold:

    1. Extract P(REAL) and P(FAKE) from the model.
    2. Compute gap = |P(REAL) – P(FAKE)|.
    3. Verdict:
       • gap >= high_thresh  → winning label wins  (high confidence)
       • low_thresh <= gap < high_thresh → SUSPICIOUS  (uncertain)
       • gap < low_thresh    → SUSPICIOUS  (model nearly 50/50)

    This avoids the original bug where a single 0.60 cut-off treated
    0.60 vs 0.60 (impossible, but near-ties like 0.61 vs 0.39) the same
    as 0.99 vs 0.01, producing overconfident verdicts on borderline text.
    """
    raw = clf(text)[0]                           # list of {'label':…,'score':…}
    scores = {r["label"]: r["score"] for r in raw}

    # RoBERTa labels from config.json: 0→FAKE, 1→REAL
    fake_prob = scores.get("FAKE", scores.get("LABEL_0", 0.0))
    real_prob = scores.get("REAL", scores.get("LABEL_1", 0.0))

    # Normalise (they should already sum to 1, but guard rounding)
    total = fake_prob + real_prob
    if total > 0:
        fake_prob /= total
        real_prob /= total

    gap          = abs(real_prob - fake_prob)
    winning_prob = max(real_prob, fake_prob)
    winning_lbl  = "REAL" if real_prob >= fake_prob else "FAKE"

    # ── Decision rule ──────────────────────────────────────────────────
    if gap >= high_thresh:
        label = f"{'REAL NEWS' if winning_lbl == 'REAL' else 'FAKE NEWS'}"
        css   = "real" if winning_lbl == "REAL" else "fake"
        conf  = "High"
    elif gap >= low_thresh:
        label = "SUSPICIOUS"
        css   = "sus"
        conf  = "Moderate"
    else:
        # Nearly 50/50 – genuinely ambiguous
        label = "SUSPICIOUS"
        css   = "sus"
        conf  = "Low"

    # Human-readable confidence %  (of the winning side)
    confidence_pct = round(winning_prob * 100, 1)

    return {
        "label":          label,
        "css":            css,
        "real_prob":      real_prob,
        "fake_prob":      fake_prob,
        "gap":            gap,
        "confidence_lvl": conf,
        "confidence_pct": confidence_pct,
        "winning_lbl":    winning_lbl,
    }


# ──────────────────────────────────────────────
# API HELPERS
# ──────────────────────────────────────────────
def extract_keywords(text: str, n: int = 5) -> str:
    """
    Very lightweight keyword extractor: strip stopwords, take top-n
    longest tokens as a search query (no external deps needed).
    """
    stopwords = {
        "the","a","an","and","or","but","in","on","at","to","for","of",
        "with","by","from","is","was","are","were","be","been","being",
        "have","has","had","do","does","did","will","would","could","should",
        "may","might","this","that","these","those","it","its","they","their",
        "we","our","you","your","he","she","him","her","his","i","my","me",
        "not","no","nor","so","yet","both","either","whether","if","as","than",
        "then","when","where","who","which","what","how","all","any","each",
        "more","most","other","some","such","only","own","same","too","very",
        "just","because","while","about","against","between","into","through",
        "during","before","after","above","below","up","down","out","off","over",
        "under","again","further","also","said","says","say",
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in stopwords]

    # score by frequency × length
    freq: dict = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq, key=lambda w: freq[w] * len(w), reverse=True)
    return " ".join(ranked[:n])


def fetch_newsapi(query: str, api_key: str) -> list[dict]:
    if not api_key:
        return []
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q":        query,
            "sortBy":   "relevancy",
            "pageSize": 4,
            "language": "en",
            "apiKey":   api_key,
        }
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        articles = data.get("articles", [])
        return [
            {
                "title":  a.get("title", "No title"),
                "source": a.get("source", {}).get("name", "Unknown"),
                "url":    a.get("url", "#"),
                "from":   "NewsAPI",
            }
            for a in articles
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]
    except Exception:
        return []


def fetch_gnews(query: str, api_key: str) -> list[dict]:
    if not api_key:
        return []
    try:
        url = "https://gnews.io/api/v4/search"
        params = {
            "q":      query,
            "token":  api_key,
            "lang":   "en",
            "max":    4,
        }
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        articles = data.get("articles", [])
        return [
            {
                "title":  a.get("title", "No title"),
                "source": a.get("source", {}).get("name", "Unknown"),
                "url":    a.get("url", "#"),
                "from":   "GNews",
            }
            for a in articles
        ]
    except Exception:
        return []


def dedupe_articles(articles: list[dict]) -> list[dict]:
    seen, out = set(), []
    for a in articles:
        key = a["title"].lower()[:60]
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


# ──────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────
st.title("Fake News Detector")
st.caption("Powered by fine-tuned RoBERTa · NLP & APIs · Final Year Project")
st.markdown("---")

# ── Load model ────────────────────────────────
if not os.path.isdir(model_path):
    st.error(
        f"Model folder not found at `{model_path}`.\n\n"
        "Please update the path in the script, or extract your "
        "`fake_news_model/` folder next to this script."
    )
    st.stop()

with st.spinner("Loading model…"):
    clf = load_model(model_path)
st.success("Model loaded successfully!")

# ── Input section ─────────────────────────────
st.subheader("Enter News Article")
input_mode = st.radio(
    "Input mode",
    ["Paste text", "Paste URL"],
    horizontal=True,
    label_visibility="collapsed",
)

article_text = ""
url_input    = ""

if input_mode == "Paste text":
    article_text = st.text_area(
        "Article text",
        height=220,
        placeholder="Paste the full article or headline here…",
        label_visibility="collapsed",
    )
else:
    url_input = st.text_input(
        "Article URL",
        placeholder="https://example.com/some-news-article",
        label_visibility="collapsed",
    )
    if url_input:
        with st.spinner("Fetching article from URL…"):
            try:
                from urllib.request import urlopen
                from html.parser import HTMLParser

                class _TextExtractor(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self._text, self._skip = [], False
                    def handle_starttag(self, tag, attrs):
                        if tag in ("script", "style", "nav", "footer", "header"):
                            self._skip = True
                    def handle_endtag(self, tag):
                        if tag in ("script", "style", "nav", "footer", "header"):
                            self._skip = False
                    def handle_data(self, data):
                        if not self._skip:
                            self._text.append(data)
                    def get_text(self):
                        return " ".join(self._text).strip()

                resp = urlopen(url_input, timeout=10)
                html = resp.read().decode("utf-8", errors="replace")
                parser = _TextExtractor()
                parser.feed(html)
                article_text = parser.get_text()[:3000]
                st.info(f"Fetched ~{len(article_text)} characters from URL.")
            except Exception as e:
                st.error(f"Could not fetch URL: {e}")

# ── Buttons ───────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    analyse_btn = st.button("Analyse Article", use_container_width=True, type="primary")
with col2:
    reset_btn = st.button("Reset", use_container_width=True)

# ── Reset ─────────────────────────────────────
if reset_btn:
    st.session_state.pop("result", None)
    st.session_state.pop("articles", None)
    st.rerun()

# ── Analysis ──────────────────────────────────
if analyse_btn:
    text = article_text.strip()
    if not text:
        st.warning("Please enter or fetch some article text first.")
    elif len(text.split()) < 5:
        st.warning("Text too short — please provide at least a full sentence.")
    else:
        with st.spinner("Analysing article…"):
            result   = classify_news(text, clf, high_conf, low_conf)
            keywords = extract_keywords(text)
            # Parallel-fetch from both APIs
            news_articles = fetch_newsapi(keywords, news_api_key)
            gnews_articles = fetch_gnews(keywords, gnews_api_key)
            all_articles  = dedupe_articles(news_articles + gnews_articles)[:6]

        st.session_state["result"]   = result
        st.session_state["articles"] = all_articles
        st.session_state["keywords"] = keywords

# ── Show result ───────────────────────────────
if "result" in st.session_state:
    result   = st.session_state["result"]
    articles = st.session_state.get("articles", [])
    keywords = st.session_state.get("keywords", "")

    st.markdown("---")
    st.subheader("Analysis Result")

    # Verdict badge
    badge_cls = {
        "real": "verdict-real",
        "fake": "verdict-fake",
        "sus":  "verdict-sus",
    }[result["css"]]
    st.markdown(
        f'<span class="verdict {badge_cls}">{result["label"]}</span> '
        f'<span class="api-badge">Confidence: {result["confidence_pct"]}% ({result["confidence_lvl"]})</span>',
        unsafe_allow_html=True,
    )


    # Probability bars
    st.markdown("**Model probability scores**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="bar-label">REAL</div>', unsafe_allow_html=True)
        st.progress(result["real_prob"])
        st.caption(f"{result['real_prob']*100:.1f}%")
    with c2:
        st.markdown('<div class="bar-label">FAKE</div>', unsafe_allow_html=True)
        st.progress(result["fake_prob"])
        st.caption(f"{result['fake_prob']*100:.1f}%")

    # ── Related articles from APIs ─────────────────
    st.markdown("---")
    if articles:
        st.subheader("Related Articles (API Cross-Check)")
        st.caption(f"Search keywords: `{keywords}`")
        for a in articles:
            badge = f'<span class="api-badge">{a["from"]}</span>'
            pill  = f'<span class="source-pill">{a["source"]}</span>'
            st.markdown(
                f'{pill}{badge}<br>'
                f'<a href="{a["url"]}" target="_blank"><b>{a["title"]}</b></a>',
                unsafe_allow_html=True,
            )
            st.markdown("")
    else:
        if news_api_key or gnews_api_key:
            st.info("No related articles found — try a different article or check your API keys.")
        else:
            st.info("Add NewsAPI or GNews keys in the code to see related articles for cross-checking.")

    # ── Disclaimer ────────────────────────────────
    st.markdown("---")
    st.warning(
        "**Disclaimer:** This tool is an AI-based aid, not a fact-checking authority. "
        "Always verify news through multiple reputable sources."
    )

    # ── Reset CTA ─────────────────────────────────
    st.markdown("")
    if st.button("Analyse another article", use_container_width=True):
        st.session_state.pop("result", None)
        st.session_state.pop("articles", None)
        st.session_state.pop("keywords", None)
        st.rerun()
