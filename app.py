"""
TruthLens — Fake News Detector
Streamlit app — works with any HuggingFace fake-news classifier.
No external APIs required. Deployable on Streamlit Community Cloud.

requirements.txt:
    streamlit>=1.32
    transformers>=4.38
    torch>=2.0
    sentencepiece
    plotly
    pandas
"""

import re
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
import torch

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens — Fake News Detector",
    page_icon="🔍",
    layout="centered",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.verdict-banner {
    padding: 1.2rem 1.6rem;
    border-radius: 12px;
    margin: 0.8rem 0 1.2rem 0;
    border-left: 6px solid;
    font-size: 1.05rem;
    line-height: 1.7;
}
.banner-real      { background:#e8faf0; border-color:#27ae60; color:#1a5c33; }
.banner-fake      { background:#fdf0ef; border-color:#e74c3c; color:#7b1c12; }
.banner-uncertain { background:#fdf8e8; border-color:#f39c12; color:#7d5a00; }

.big-label {
    display: inline-block;
    font-size: 1.5rem;
    font-weight: 700;
    padding: 0.3rem 1.2rem;
    border-radius: 999px;
    margin-bottom: 0.8rem;
}
.label-real      { background:#27ae60; color:#fff; }
.label-fake      { background:#e74c3c; color:#fff; }
.label-uncertain { background:#f39c12; color:#fff; }

.signal-row { display:flex; gap:0.6rem; flex-wrap:wrap; margin:0.4rem 0 1rem 0; }
.signal-pill {
    display:inline-block; padding:0.25rem 0.75rem;
    border-radius:999px; font-size:0.78rem; font-weight:600; border:1px solid;
}
.pill-green  { background:#f0fdf4; color:#166534; border-color:#bbf7d0; }
.pill-red    { background:#fef2f2; color:#991b1b; border-color:#fecaca; }
.pill-grey   { background:#f8fafc; color:#64748b; border-color:#cbd5e1; }

.bar-wrap { margin-bottom: 1rem; }
.bar-label-row {
    display:flex; justify-content:space-between;
    font-size:0.85rem; font-weight:600; margin-bottom:0.2rem;
}
.history-item {
    display:flex; align-items:center; gap:0.8rem;
    padding:0.5rem 0; border-bottom:1px solid #f1f5f9; font-size:0.83rem;
}
.history-verdict {
    font-weight:700; font-size:0.75rem;
    padding:2px 8px; border-radius:999px; white-space:nowrap;
}
.hv-real      { background:#27ae60; color:#fff; }
.hv-fake      { background:#e74c3c; color:#fff; }
.hv-uncertain { background:#f39c12; color:#fff; }
div[data-testid="stButton"] button { border-radius:8px !important; font-weight:600 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SETTINGS
# Change MODEL_ID to your own HuggingFace model after you push it:
#   from transformers import AutoModelForSequenceClassification, AutoTokenizer
#   model.push_to_hub("your-username/your-model-name")
#   tokenizer.push_to_hub("your-username/your-model-name")
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ID        = "iamsahhil/fakenews"
FAKE_THRESHOLD  = 0.60
REAL_THRESHOLD  = 0.60

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
for key, default in [("input_text",""), ("result",None), ("history",[])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADER — cached for entire session
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSequenceClassification.from_pretrained(model_id)
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
    return clf, model.config

# ──────────────────────────────────────────────────────────────────────────────
# LABEL NORMALISER
# Handles every label convention seen in public fake-news HF models:
#   FAKE/REAL, fake/real, LABEL_0/LABEL_1, 0/1, MISLEADING/CREDIBLE
# ──────────────────────────────────────────────────────────────────────────────
def extract_fake_real(raw_scores: list, config) -> tuple:
    scores = {r["label"].upper(): r["score"] for r in raw_scores}

    if "FAKE" in scores and "REAL" in scores:
        return scores["FAKE"], scores["REAL"]

    if "MISLEADING" in scores and "CREDIBLE" in scores:
        return scores["MISLEADING"], scores["CREDIBLE"]

    if "FAKE" in scores:
        return scores["FAKE"], 1.0 - scores["FAKE"]

    if "REAL" in scores:
        return 1.0 - scores["REAL"], scores["REAL"]

    if "LABEL_0" in scores and "LABEL_1" in scores:
        id2label = {int(k): v.upper() for k, v in config.id2label.items()}
        label_0  = id2label.get(0, "LABEL_0")
        fake_kw  = {"FAKE","FALSE","MISLEADING","MISINFORMATION","0"}
        real_kw  = {"REAL","TRUE","CREDIBLE","LEGITIMATE","1"}
        if label_0 in fake_kw:
            return scores["LABEL_0"], scores["LABEL_1"]
        if label_0 in real_kw:
            return scores["LABEL_1"], scores["LABEL_0"]
        # Most publicly uploaded fake-news models: LABEL_0=fake, LABEL_1=real
        return scores["LABEL_0"], scores["LABEL_1"]

    if len(raw_scores) == 1:
        lbl   = raw_scores[0]["label"].upper()
        score = raw_scores[0]["score"]
        if any(k in lbl for k in ["FAKE","FALSE","MISLEAD"]):
            return score, 1.0 - score
        return 1.0 - score, score

    vals = list(scores.values())
    return (vals[0], vals[1]) if len(vals) > 1 else (0.5, 0.5)

# ──────────────────────────────────────────────────────────────────────────────
# LINGUISTIC SIGNAL PATTERNS
# ──────────────────────────────────────────────────────────────────────────────
FAKE_PATTERNS = {
    "Sensationalist language":
        r"\b(shocking|breaking|exclusive|bombshell|exposed|cover.?up|hidden truth|secret)\b",
    "Extreme emotional language":
        r"\b(outrage|furious|disgusting|terrifying|unbelievable|insane|crazy|mind.?blowing)\b",
    "Vague or unverified attribution":
        r"\b(sources say|some people|many people|experts warn|scientists confirm|doctors reveal|they say)\b",
    "Call-to-action / urgency markers":
        r"\b(share this|must read|spread the word|wake up|before it.?s deleted|they don.?t want you to know)\b",
    "Conspiracy vocabulary":
        r"\b(deep state|new world order|mainstream media|fake media|psyop|plandemic|scamdemic|cabal)\b",
    "Viral forward / WhatsApp markers":
        r"(forward|plz share|please share|🚨|⚠️|🔴|❗|share before deleted)",
}

REAL_PATTERNS = {
    "Named credible news source":
        r"\b(reuters|associated press|ap news|afp|bbc|the guardian|new york times|washington post)\b",
    "Specific attributed statement":
        r"\b(according to|said in a statement|told reporters|confirmed by|announced that|said on)\b",
    "Journalistic hedging / accuracy markers":
        r"\b(however|although|despite|officials said|could not independently verify|pending confirmation)\b",
    "Specific dated claim":
        r"\b\d{4}\b.{0,60}\b(said|reported|showed|found|confirmed|announced)\b",
}

def analyse_signals(text: str) -> dict:
    lower     = text.lower()
    fake_hits = {name: pat for name, pat in FAKE_PATTERNS.items()
                 if re.search(pat, lower, re.IGNORECASE)}
    real_hits = {name: pat for name, pat in REAL_PATTERNS.items()
                 if re.search(pat, lower, re.IGNORECASE)}
    return {
        "fake_count": len(fake_hits),
        "real_count": len(real_hits),
        "fake_names": list(fake_hits.keys()),
        "real_names": list(real_hits.keys()),
    }

# ──────────────────────────────────────────────────────────────────────────────
# MAIN CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────
def classify(text: str, clf, config) -> dict:
    text = text.strip()

    # Step 1 — model on full text
    raw  = clf(text[:1024])[0]
    fp, rp = extract_fake_real(raw, config)

    # Step 2 — headline ensemble for longer texts
    sentences = re.split(r'(?<=[.!?])\s+', text)
    headline  = sentences[0][:512] if sentences else text[:512]
    is_long   = len(text.split()) >= 15
    ensemble  = is_long and len(headline) > 20

    if ensemble:
        raw_h     = clf(headline)[0]
        fp_h, rp_h = extract_fake_real(raw_h, config)
        fp = 0.70 * fp + 0.30 * fp_h
        rp = 0.70 * rp + 0.30 * rp_h

    # Renormalise
    t = fp + rp
    if t > 0:
        fp /= t; rp /= t

    # Step 3 — linguistic nudge (capped at 15%)
    signals    = analyse_signals(text)
    fp = min(fp + signals["fake_count"] * 0.04, 0.97)
    rp = min(rp + signals["real_count"] * 0.03, 0.97)
    t  = fp + rp
    if t > 0:
        fp /= t; rp /= t

    # Step 4 — verdict
    conf = max(fp, rp)
    if fp >= FAKE_THRESHOLD:
        verdict, css = "FAKE NEWS",  "fake"
    elif rp >= REAL_THRESHOLD:
        verdict, css = "REAL NEWS",  "real"
    else:
        verdict, css = "UNCERTAIN",  "uncertain"

    return {
        "verdict":    verdict,
        "css":        css,
        "fake_prob":  round(fp,       4),
        "real_prob":  round(rp,       4),
        "confidence": round(conf*100, 1),
        "signals":    signals,
        "ensemble":   ensemble,
    }

# ──────────────────────────────────────────────────────────────────────────────
# RESULT RENDERER
# ──────────────────────────────────────────────────────────────────────────────
ICONS = {"FAKE NEWS":"❌","REAL NEWS":"✅","UNCERTAIN":"⚠️"}

def render_result(r: dict):
    css  = r["css"]
    icon = ICONS.get(r["verdict"],"⚠️")
    lc   = f"label-{css}"
    expl = {
        "fake":      "Strong indicators of fabricated or misleading content detected.",
        "real":      "Content consistent with genuine, credibly-written reporting.",
        "uncertain": "The model is not confident. Verify with trusted sources before sharing.",
    }[css]

    st.markdown(f'<div class="big-label {lc}">{icon} {r["verdict"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="verdict-banner banner-{css}">'
        f'<strong>Confidence: {r["confidence"]}%</strong><br>{expl}</div>',
        unsafe_allow_html=True
    )

    st.markdown("#### Probability Scores")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="bar-label-row"><span>❌ FAKE</span><span>{r["fake_prob"]*100:.1f}%</span></div>', unsafe_allow_html=True)
        st.progress(float(r["fake_prob"]))
    with c2:
        st.markdown(f'<div class="bar-label-row"><span>✅ REAL</span><span>{r["real_prob"]*100:.1f}%</span></div>', unsafe_allow_html=True)
        st.progress(float(r["real_prob"]))

    # Signal pills
    pills = []
    if r["ensemble"]:
        pills.append('<span class="signal-pill pill-grey">Headline + Body Ensemble</span>')
    if r["signals"]["fake_count"]:
        n = r["signals"]["fake_count"]
        pills.append(f'<span class="signal-pill pill-red">⚠ {n} fake signal{"s" if n>1 else ""}</span>')
    if r["signals"]["real_count"]:
        n = r["signals"]["real_count"]
        pills.append(f'<span class="signal-pill pill-green">✓ {n} credible signal{"s" if n>1 else ""}</span>')
    if pills:
        st.markdown('<div class="signal-row">' + "".join(pills) + '</div>', unsafe_allow_html=True)

    # Signal detail
    with st.expander("🔎 Linguistic Signal Detail", expanded=False):
        if r["signals"]["fake_names"]:
            st.markdown("**Fake-news patterns found:**")
            for name in r["signals"]["fake_names"]:
                st.markdown(f"- {name}")
        if r["signals"]["real_names"]:
            st.markdown("**Credible-reporting patterns found:**")
            for name in r["signals"]["real_names"]:
                st.markdown(f"- {name}")
        if not r["signals"]["fake_names"] and not r["signals"]["real_names"]:
            st.info("No strong linguistic signals. Verdict based entirely on the ML model.")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("🔍 TruthLens")
st.caption("Fake news detector · Fine-tuned RoBERTa + Linguistic Analysis · No API keys needed")
st.markdown("---")

with st.spinner("Loading model from HuggingFace Hub…"):
    try:
        clf, config = load_model(MODEL_ID)
        st.success(f"✅ Model ready — `{MODEL_ID}`")
    except Exception as e:
        st.error(f"❌ Could not load model `{MODEL_ID}`: {e}")
        st.stop()

st.markdown("---")

tab_single, tab_batch, tab_history = st.tabs(
    ["🔍 Verify Claim", "📋 Batch Check", "📊 History"]
)

# ── SINGLE ────────────────────────────────────────────────────────────────────
with tab_single:
    EXAMPLES = [
        "",
        "India gained independence on August 15, 1947 after centuries of British colonial rule.",
        "URGENT: Drinking cow urine mixed with turmeric cures cancer in 7 days — doctors hiding this!",
        "NASA confirms the Moon is a hollow alien spaceship placed in orbit 4000 years ago.",
        "The Federal Reserve held interest rates steady at its latest policy meeting.",
        "5G towers are secretly programmed to spread COVID-19 on government orders — SHARE NOW!",
        "Scientists have developed a solar cell with 47% efficiency using perovskite materials.",
    ]

    ex = st.selectbox("Try an example →", EXAMPLES, label_visibility="collapsed")
    if ex and ex != st.session_state.input_text:
        st.session_state.input_text = ex

    article_text = st.text_area(
        "Enter a news headline or article",
        value=st.session_state.input_text,
        height=180,
        placeholder="Paste a headline, WhatsApp forward, or full article here…",
        key="single_input",
    )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        analyse_btn = st.button("🔍 Analyse", use_container_width=True, type="primary",
                                 disabled=not article_text.strip())
    with col_b:
        reset_btn = st.button("↺ Reset", use_container_width=True)

    # ── RESET — correct pattern: clear state THEN rerun ───────────────────────
    if reset_btn:
        st.session_state.input_text = ""
        st.session_state.result     = None
        st.rerun()

    if analyse_btn and article_text.strip():
        st.session_state.input_text = article_text.strip()
        with st.spinner("Analysing…"):
            result = classify(article_text.strip(), clf, config)
        st.session_state.result = result
        css = result["css"]
        st.session_state.history.insert(0, {
            "text":       article_text.strip()[:80],
            "verdict":    result["verdict"],
            "confidence": result["confidence"],
            "css":        css,
        })
        st.session_state.history = st.session_state.history[:50]

    if st.session_state.result:
        st.markdown("---")
        render_result(st.session_state.result)

# ── BATCH ─────────────────────────────────────────────────────────────────────
with tab_batch:
    st.caption("One claim per line. Results download as CSV.")
    batch_text = st.text_area(
        "Claims", height=200,
        placeholder="India gained independence in 1947.\n5G towers spread COVID-19.\nCow urine cures cancer.",
        label_visibility="collapsed", key="batch_input",
    )
    run_batch = st.button("▶ Run Batch", type="primary", disabled=not batch_text.strip())

    if run_batch and batch_text.strip():
        import pandas as pd
        claims = [c.strip() for c in batch_text.splitlines() if c.strip()]
        prog, status = st.progress(0), st.empty()
        rows = []
        for i, claim in enumerate(claims):
            status.caption(f"Checking {i+1}/{len(claims)}: {claim[:60]}…")
            r = classify(claim, clf, config)
            rows.append({
                "Claim":          claim[:100],
                "Verdict":        r["verdict"],
                "Confidence (%)": r["confidence"],
                "Fake Prob (%)":  round(r["fake_prob"]*100, 1),
                "Real Prob (%)":  round(r["real_prob"]*100, 1),
                "Fake Signals":   r["signals"]["fake_count"],
                "Real Signals":   r["signals"]["real_count"],
            })
            prog.progress((i+1)/len(claims))

        status.empty(); prog.empty()
        df = pd.DataFrame(rows)

        def colour_v(val):
            c = {"FAKE NEWS":"#e74c3c","REAL NEWS":"#27ae60","UNCERTAIN":"#f39c12"}.get(val,"#888")
            return f"color:{c};font-weight:bold"

        st.dataframe(df.style.applymap(colour_v, subset=["Verdict"]),
                     use_container_width=True, height=360)

        vc = df["Verdict"].value_counts()
        c1,c2,c3 = st.columns(3)
        c1.metric("✅ REAL",      vc.get("REAL NEWS", 0))
        c2.metric("❌ FAKE",      vc.get("FAKE NEWS", 0))
        c3.metric("⚠️ UNCERTAIN", vc.get("UNCERTAIN", 0))
        st.download_button("⬇ Download CSV", df.to_csv(index=False).encode(),
                            "truthlens_results.csv", "text/csv")

        for row in rows:
            css = {"FAKE NEWS":"fake","REAL NEWS":"real","UNCERTAIN":"uncertain"}.get(row["Verdict"],"uncertain")
            st.session_state.history.insert(0, {
                "text": row["Claim"][:80], "verdict": row["Verdict"],
                "confidence": row["Confidence (%)"], "css": css,
            })
        st.session_state.history = st.session_state.history[:50]

# ── HISTORY ───────────────────────────────────────────────────────────────────
with tab_history:
    hist = st.session_state.history
    if not hist:
        st.markdown("<div style='text-align:center;color:#94a3b8;margin-top:2rem;'>No checks yet.</div>",
                    unsafe_allow_html=True)
    else:
        hvc   = {"real":"hv-real","fake":"hv-fake","uncertain":"hv-uncertain"}
        icons = {"REAL NEWS":"✅","FAKE NEWS":"❌","UNCERTAIN":"⚠️"}
        for h in hist[:30]:
            st.markdown(
                f'<div class="history-item">'
                f'<span>{icons.get(h["verdict"],"⚠️")}</span>'
                f'<span style="flex:1;color:#374151;">{h["text"]}{"…" if len(h["text"])>=80 else ""}</span>'
                f'<span class="history-verdict {hvc.get(h["css"],"hv-uncertain")}">{h["verdict"]}</span>'
                f'<span style="color:#9ca3af;font-size:0.75rem;min-width:42px;text-align:right;">{h["confidence"]}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("---")
        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()

        if len(hist) >= 3:
            import pandas as pd
            vc  = pd.Series([h["verdict"] for h in hist]).value_counts()
            clr = {"REAL NEWS":"#27ae60","FAKE NEWS":"#e74c3c","UNCERTAIN":"#f39c12"}
            try:
                import plotly.graph_objects as go
                fig = go.Figure(go.Pie(
                    labels=vc.index, values=vc.values,
                    marker_colors=[clr.get(v,"#888") for v in vc.index],
                    hole=0.60, textfont=dict(size=12),
                ))
                fig.update_layout(
                    height=220, margin=dict(t=10,b=10,l=10,r=10),
                    paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
                    annotations=[dict(text=f"<b>{len(hist)}</b><br>checks",
                                       showarrow=False, font=dict(size=16))],
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(vc)

# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.warning("⚠️ TruthLens is AI-assisted. Always verify important claims with multiple trusted sources.")
