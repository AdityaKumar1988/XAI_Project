import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import pytesseract
from PIL import Image
from modules.preprocessor import DocumentPreprocessor
from modules.segmentor import RegionSegmentor
from modules.classifier import DocumentClassifier
from modules.decision import DecisionEngine

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocVerify – Fake Document Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Tokens ── */
:root {
    --bg-base:      #0d0f14;
    --bg-surface:   #13161e;
    --bg-raised:    #1a1e2a;
    --bg-hover:     #1f2535;
    --border:       rgba(255,255,255,0.07);
    --border-lit:   rgba(0,210,170,0.35);
    --accent:       #00d2aa;
    --accent-dim:   rgba(0,210,170,0.12);
    --accent-glow:  rgba(0,210,170,0.25);
    --danger:       #ff4d6d;
    --danger-dim:   rgba(255,77,109,0.12);
    --warning:      #f59e0b;
    --warning-dim:  rgba(245,158,11,0.12);
    --success:      #00d2aa;
    --success-dim:  rgba(0,210,170,0.12);
    --text-primary: #f0f2f8;
    --text-secondary: #8b91a7;
    --text-muted:   #4e5568;
    --radius-sm:    6px;
    --radius-md:    12px;
    --radius-lg:    18px;
    --shadow-sm:    0 2px 8px rgba(0,0,0,0.4);
    --shadow-md:    0 8px 32px rgba(0,0,0,0.5);
    --shadow-lg:    0 16px 64px rgba(0,0,0,0.6);
}

/* ── Base ── */
html, body, .stApp {
    background-color: var(--bg-base) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Subtle dot-grid background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(circle, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 28px 28px;
    pointer-events: none;
    z-index: 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--bg-raised); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-dim); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

/* ── Sidebar selectbox & slider labels ── */
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    transition: border-color 0.2s;
}
.stSelectbox > div > div:hover {
    border-color: var(--border-lit) !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: var(--accent) !important;
}
.stSlider > div > div > div {
    background: var(--bg-raised) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-surface) !important;
    border: 1.5px dashed var(--border-lit) !important;
    border-radius: var(--radius-lg) !important;
    padding: 2rem 1.5rem !important;
    transition: background 0.25s, border-color 0.25s;
}
[data-testid="stFileUploader"]:hover {
    background: var(--accent-dim) !important;
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: var(--text-secondary) !important;
    font-size: 0.88rem !important;
}

/* ── Primary Button ── */
.stButton > button {
    background: var(--accent) !important;
    color: #050708 !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em;
    padding: 0.65rem 2rem !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 0 0 0 var(--accent-glow);
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 0 24px var(--accent-glow) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    opacity: 1 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--accent) !important;
}

/* ── Progress Bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent), #00b894) !important;
    border-radius: 99px !important;
}
.stProgress > div > div > div {
    background: var(--bg-raised) !important;
    border-radius: 99px !important;
    height: 6px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.streamlit-expanderContent {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ── Alert / Error ── */
.stAlert {
    background: var(--bg-raised) !important;
    border-radius: var(--radius-sm) !important;
    border-left: 3px solid var(--danger) !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1300px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.dv-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 2rem 0 2.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2.5rem;
}
.dv-header-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #00d2aa22, #00d2aa44);
    border: 1px solid #00d2aa55;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 0 24px #00d2aa22;
}
.dv-header-text h1 {
    margin: 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #f0f2f8;
    letter-spacing: -0.01em;
}
.dv-header-text p {
    margin: 0.15rem 0 0 0;
    font-size: 0.82rem;
    color: #8b91a7;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.dv-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0,210,170,0.1);
    border: 1px solid rgba(0,210,170,0.25);
    border-radius: 99px;
    padding: 0.3rem 0.75rem;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: #00d2aa;
    letter-spacing: 0.04em;
    margin-left: auto;
}
.dv-pill-dot {
    width: 6px; height: 6px;
    background: #00d2aa;
    border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
</style>
<div class="dv-header">
    <div class="dv-header-icon">🛡️</div>
    <div class="dv-header-text">
        <h1>DocVerify</h1>
        <p>Academic Document Forensic Analysis</p>
    </div>
    <div class="dv-pill">
        <div class="dv-pill-dot"></div>
        SYSTEM READY
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-section {
        margin-bottom: 2rem;
    }
    .sidebar-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #4e5568;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .stat-chip {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #1a1e2a;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.5rem;
    }
    .stat-chip span:first-child {
        font-size: 0.75rem;
        color: #8b91a7;
        text-transform: none !important;
        letter-spacing: 0 !important;
    }
    .stat-chip span:last-child {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #00d2aa;
    }
    </style>
    <div class="sidebar-label">Configuration</div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Detection Engine",
        ["CNN (Deep Learning)", "Random Forest (ML)"],
        help="Select the underlying model for analysis"
    )
    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.01,
        help="Minimum confidence to flag a document as FAKE"
    )

    st.markdown("""
    <div style="height:1.5rem"></div>
    <div class="sidebar-label">System Info</div>
    <div class="stat-chip">
        <span>Model Status</span>
        <span>● Online</span>
    </div>
    <div class="stat-chip">
        <span>OCR Engine</span>
        <span>Tesseract</span>
    </div>
    <div class="stat-chip">
        <span>Framework</span>
        <span>OpenCV + ML</span>
    </div>
    """, unsafe_allow_html=True)


# ── Load Components ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_components():
    return DocumentPreprocessor(), RegionSegmentor(), DocumentClassifier(), DecisionEngine()

try:
    preproc, seg, clf, engine = load_components()
except Exception as e:
    st.error(f"Failed to initialise modules: {e}")
    st.stop()


# ── Upload Area ─────────────────────────────────────────────────────────────────
st.markdown("""
<p style="font-size:0.75rem;color:#4e5568;text-transform:uppercase;
   letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;margin-bottom:0.6rem;">
  Input Document
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop a document image here, or click to browse",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed",
)


# ── Main Content ────────────────────────────────────────────────────────────────
if uploaded_file is not None:

    # Save to temp
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tfile.write(uploaded_file.read())
    tfile.close()

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    col_img, col_gap, col_result = st.columns([5, 0.3, 5])

    with col_img:
        st.markdown("""
        <p style="font-size:0.75rem;color:#4e5568;text-transform:uppercase;
           letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;margin-bottom:0.6rem;">
          Original Document
        </p>
        """, unsafe_allow_html=True)
        st.image(uploaded_file, use_column_width=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        analyze = st.button("⬡  Run Forensic Analysis", key="analyze_btn")

    # ── Analysis ──────────────────────────────────────────────────────────────
    if analyze:
        with col_result:
            with st.spinner("Running forensic pipeline…"):
                try:
                    img_original = cv2.imread(tfile.name)
                    img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
                    img_gray = preproc.preprocess(tfile.name)
                    rois, annotated_seg = seg.segment(img_gray)

                    # OCR
                    extracted_text = ""
                    try:
                        extracted_text = pytesseract.image_to_string(img_gray)
                    except Exception:
                        pass

                    # Feature extraction (both branches use RF currently)
                    heatmap = None
                    from modules.feature_extractor import FeatureExtractor
                    feat = FeatureExtractor()
                    if not rois:
                        rois = [(img_gray, (0, 0, img_gray.shape[1], img_gray.shape[0]))]
                    vector = feat.extract_all(rois)
                    label, conf = clf.predict_rf(vector)

                    # Threshold override
                    if conf < threshold and label == 1:
                        decision = "UNCERTAIN"
                    else:
                        decision = engine.make_decision(label, conf)

                    # ── Result Badge ────────────────────────────────────────
                    if decision == "FAKE":
                        badge_bg  = "linear-gradient(135deg,#ff4d6d18,#ff4d6d10)"
                        badge_bdr = "#ff4d6d55"
                        badge_col = "#ff4d6d"
                        badge_icon = "✕"
                        badge_sub  = "Document shows signs of tampering"
                    elif "UNCERTAIN" in decision:
                        badge_bg  = "linear-gradient(135deg,#f59e0b18,#f59e0b10)"
                        badge_bdr = "#f59e0b55"
                        badge_col = "#f59e0b"
                        badge_icon = "◎"
                        badge_sub  = "Manual review recommended"
                    else:
                        badge_bg  = "linear-gradient(135deg,#00d2aa18,#00d2aa10)"
                        badge_bdr = "#00d2aa55"
                        badge_col = "#00d2aa"
                        badge_icon = "✓"
                        badge_sub  = "Document appears authentic"

                    conf_pct = f"{conf * 100:.1f}"

                    st.markdown(f"""
                    <style>
                    @keyframes badge-in {{
                        from {{ opacity:0; transform:translateY(10px); }}
                        to   {{ opacity:1; transform:translateY(0); }}
                    }}
                    .result-badge {{
                        background: {badge_bg};
                        border: 1px solid {badge_bdr};
                        border-radius: 16px;
                        padding: 1.75rem 2rem;
                        margin-bottom: 1.25rem;
                        animation: badge-in 0.4s ease;
                    }}
                    .result-badge-header {{
                        display: flex;
                        align-items: center;
                        gap: 0.75rem;
                        margin-bottom: 0.5rem;
                    }}
                    .result-icon {{
                        width: 36px; height: 36px;
                        background: {badge_col}22;
                        border: 1px solid {badge_col}44;
                        border-radius: 8px;
                        display: flex; align-items: center; justify-content: center;
                        font-size: 1rem;
                        color: {badge_col};
                        font-weight: 700;
                    }}
                    .result-verdict {{
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 1.4rem;
                        font-weight: 600;
                        color: {badge_col};
                        letter-spacing: 0.04em;
                    }}
                    .result-sub {{
                        font-size: 0.8rem;
                        color: #8b91a7;
                        margin-top: 0.35rem;
                    }}
                    .conf-row {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-top: 1rem;
                        padding-top: 1rem;
                        border-top: 1px solid rgba(255,255,255,0.06);
                    }}
                    .conf-label {{
                        font-size: 0.72rem;
                        color: #4e5568;
                        text-transform: uppercase;
                        letter-spacing: 0.08em;
                        font-family: 'JetBrains Mono', monospace;
                    }}
                    .conf-value {{
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 1.1rem;
                        font-weight: 600;
                        color: {badge_col};
                    }}
                    </style>

                    <div class="result-badge">
                        <div class="result-badge-header">
                            <div class="result-icon">{badge_icon}</div>
                            <div>
                                <div class="result-verdict">{decision}</div>
                                <div class="result-sub">{badge_sub}</div>
                            </div>
                        </div>
                        <div class="conf-row">
                            <span class="conf-label">Confidence Score</span>
                            <span class="conf-value">{conf_pct}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.progress(float(conf))

                    # ── Segmentation ─────────────────────────────────────────
                    st.markdown("""
                    <p style="font-size:0.75rem;color:#4e5568;text-transform:uppercase;
                       letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;
                       margin:1.25rem 0 0.5rem 0;">
                      Region Analysis
                    </p>
                    """, unsafe_allow_html=True)
                    st.image(annotated_seg, use_column_width=True)

                    # ── Heatmap ──────────────────────────────────────────────
                    if heatmap is not None:
                        st.markdown("""
                        <p style="font-size:0.75rem;color:#4e5568;text-transform:uppercase;
                           letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;
                           margin:1.25rem 0 0.5rem 0;">
                          Grad-CAM Heatmap
                        </p>
                        """, unsafe_allow_html=True)
                        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        st.image(heatmap_rgb, use_column_width=True)

                    # ── OCR Text ─────────────────────────────────────────────
                    if extracted_text.strip():
                        with st.expander("Extracted OCR Text"):
                            st.markdown(f"""
                            <pre style="font-family:'JetBrains Mono',monospace;
                                        font-size:0.75rem;
                                        color:#8b91a7;
                                        white-space:pre-wrap;
                                        line-height:1.6;">{extracted_text}</pre>
                            """, unsafe_allow_html=True)

                    # ── Success toast ─────────────────────────────────────────
                    st.markdown("""
                    <div style="display:flex;align-items:center;gap:0.5rem;
                                background:#00d2aa0d;border:1px solid #00d2aa22;
                                border-radius:8px;padding:0.65rem 1rem;margin-top:1rem;">
                        <span style="color:#00d2aa;font-size:0.8rem;">✓</span>
                        <span style="color:#8b91a7;font-size:0.8rem;
                                     font-family:'JetBrains Mono',monospace;">
                            Analysis pipeline complete
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f"""
                    <div style="background:#ff4d6d0d;border:1px solid #ff4d6d33;
                                border-radius:10px;padding:1.25rem 1.5rem;">
                        <p style="color:#ff4d6d;font-family:'JetBrains Mono',monospace;
                                  font-size:0.8rem;margin:0 0 0.35rem 0;">
                            ANALYSIS FAILED
                        </p>
                        <p style="color:#8b91a7;font-size:0.82rem;margin:0;">
                            {e}. Ensure models are trained and the dataset is valid.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

    os.unlink(tfile.name)

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 2rem;
        text-align: center;
        border: 1px dashed rgba(255,255,255,0.07);
        border-radius: 18px;
        background: linear-gradient(135deg,#13161e,#1a1e2a);
        margin-top: 1.5rem;
    }
    .empty-state-icon {
        width: 64px; height: 64px;
        background: rgba(0,210,170,0.08);
        border: 1px solid rgba(0,210,170,0.2);
        border-radius: 16px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 0 40px rgba(0,210,170,0.1);
    }
    .empty-state h3 {
        font-size: 1.05rem;
        font-weight: 500;
        color: #f0f2f8;
        margin: 0 0 0.4rem 0;
    }
    .empty-state p {
        font-size: 0.82rem;
        color: #4e5568;
        margin: 0;
        max-width: 340px;
    }
    .empty-steps {
        display: flex;
        gap: 1.5rem;
        margin-top: 2.5rem;
        justify-content: center;
        flex-wrap: wrap;
    }
    .empty-step {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        background: #1a1e2a;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 99px;
        padding: 0.5rem 1rem;
    }
    .step-num {
        width: 20px; height: 20px;
        background: rgba(0,210,170,0.15);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: #00d2aa;
    }
    .step-text {
        font-size: 0.78rem;
        color: #8b91a7;
    }
    </style>
    <div class="empty-state">
        <div class="empty-state-icon">📄</div>
        <h3>No Document Loaded</h3>
        <p>Upload an academic document above to begin forensic analysis</p>
        <div class="empty-steps">
            <div class="empty-step">
                <div class="step-num">1</div>
                <span class="step-text">Upload image</span>
            </div>
            <div class="empty-step">
                <div class="step-num">2</div>
                <span class="step-text">Configure model</span>
            </div>
            <div class="empty-step">
                <div class="step-num">3</div>
                <span class="step-text">Run analysis</span>
            </div>
            <div class="empty-step">
                <div class="step-num">4</div>
                <span class="step-text">Review verdict</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)