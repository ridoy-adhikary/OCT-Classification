import base64
import html
import os
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# -----------------------------------------------------------------------------
# Configuration and model paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "..", "best_model_augmented_focal.keras"),
    os.path.join(BASE_DIR, "best_model_augmented_focal.keras"),
    os.path.join(BASE_DIR, "..", "final_eye_disease_model.keras"),
]
MODEL_PATH = next((path for path in MODEL_CANDIDATES if os.path.exists(path)), None)
CONFUSION_MATRIX_PATH = os.path.join(REPO_DIR, "confusion_matrix_focal_loss.png")
GITHUB_URL = "https://github.com/ridoy-adhikary/OCT-Classification"
DOCUMENTATION_URL = f"{GITHUB_URL}/blob/main/Project_Documentation.pdf"

NAVY = "#0F172A"
SLATE = "#334155"
MUTED = "#64748B"
BORDER = "#E2E8F0"
BG = "#F8FAFC"
TEAL = "#0D9488"
TEAL_DARK = "#0F766E"
AMBER = "#F59E0B"

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_META = {
    "CNV": {"color": "#DC2626", "soft": "#FEF2F2", "severity": "High priority"},
    "DME": {"color": "#D97706", "soft": "#FFFBEB", "severity": "Moderate priority"},
    "DRUSEN": {"color": "#B45309", "soft": "#FFFBEB", "severity": "Monitor"},
    "NORMAL": {"color": TEAL, "soft": "#F0FDFA", "severity": "Healthy"},
}
CLASS_INFO = {
    "CNV": {
        "title": "Choroidal Neovascularization",
        "description": "Abnormal blood vessel growth beneath the retina that can be associated with rapid vision changes.",
        "recommendation": "Consult a retina specialist promptly for clinical evaluation.",
    },
    "DME": {
        "title": "Diabetic Macular Edema",
        "description": "Fluid accumulation in the macula that may occur when retinal vessels leak in diabetic eye disease.",
        "recommendation": "Arrange an ophthalmology appointment and continue diabetes care with your clinician.",
    },
    "DRUSEN": {
        "title": "Drusen",
        "description": "Small deposits beneath the retina that can be associated with age-related retinal changes.",
        "recommendation": "Discuss the finding with an eye-care professional and keep regular eye appointments.",
    },
    "NORMAL": {
        "title": "Normal Retina",
        "description": "The scan does not show the image patterns associated with the three other categories in this model.",
        "recommendation": "Continue routine eye examinations as recommended by your clinician.",
    },
}


# -----------------------------------------------------------------------------
# Inline SVG helpers keep the app independent from external image assets.
# -----------------------------------------------------------------------------
def svg_icon(name, size=20, stroke="currentColor"):
    paths = {
        "eye": '<path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z"/><circle cx="12" cy="12" r="2.5"/>',
        "brain": '<path d="M9.5 4.2A3.2 3.2 0 0 0 4.3 7a3.2 3.2 0 0 0 .4 4.2A3.2 3.2 0 0 0 7 16.8a3.2 3.2 0 0 0 5 2.4 3.2 3.2 0 0 0 5-2.4 3.2 3.2 0 0 0 2.3-5.6A3.2 3.2 0 0 0 19.7 7a3.2 3.2 0 0 0-5.2-2.8A3.2 3.2 0 0 0 9.5 4.2Z"/><path d="M12 4v16M8 8.5h4M12 12h4M8 14.5h4"/>',
        "target": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"/>',
        "spark": '<path d="m12 3 1.2 5.8L19 10l-5.8 1.2L12 17l-1.2-5.8L5 10l5.8-1.2L12 3Z"/><path d="m19 16 .5 2.5L22 19l-2.5.5L19 22l-.5-2.5L16 19l2.5-.5L19 16Z"/>',
        "check": '<path d="m5 12 4 4L19 6"/>',
        "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    }
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths[name]}</svg>'


def logo_data_uri():
    logo_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
    <rect width="64" height="64" rx="16" fill="#0D9488"/>
    <path d="M10 32s8-14 22-14 22 14 22 14-8 14-22 14S10 32 10 32Z" fill="none" stroke="#fff" stroke-width="4"/>
    <circle cx="32" cy="32" r="7" fill="#fff"/>
    <circle cx="32" cy="32" r="3" fill="#0D9488"/>
    </svg>"""
    encoded = base64.b64encode(logo_svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


# -----------------------------------------------------------------------------
# Page theme and layout CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RetinaAI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {{
        --navy: {NAVY}; --slate: {SLATE}; --muted: {MUTED}; --border: {BORDER};
        --bg: {BG}; --teal: {TEAL}; --teal-dark: {TEAL_DARK}; --amber: {AMBER};
    }}
    html {{ scroll-behavior: smooth; }}
    html, body, [class*="css"] {{ font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .stApp {{ background:var(--bg); color:var(--navy); }}
    .stApp > header {{ background:transparent; }}
    [data-testid="stAppViewContainer"] {{ padding-top:0 !important; }}
    [data-testid="stHeader"] {{ display:none; }}
    [data-testid="stToolbar"], #MainMenu, footer {{ display:none !important; }}
    [data-testid="stSidebar"] {{ display:none; }}
    .block-container {{ max-width:1180px; padding:0 2rem 1rem; }}

    /* Navbar */
    .navbar {{ position:relative; z-index:10; display:grid; grid-template-columns:1fr auto 1fr; align-items:center; min-height:64px; margin:0 -2rem; padding:0 2rem; background:rgba(255,255,255,.96); border-bottom:1px solid var(--border); box-shadow:0 1px 8px rgba(15,23,42,.04); backdrop-filter:blur(12px); }}
    .brand {{ display:flex; align-items:center; gap:10px; color:var(--navy); font-size:1.08rem; font-weight:700; letter-spacing:-.02em; }}
    .brand img {{ width:34px; height:34px; border-radius:10px; }}
    .nav-links {{ display:flex; align-items:center; justify-content:center; gap:1.15rem; }}
    .nav-links a {{ color:var(--muted); font-size:.72rem; font-weight:600; text-decoration:none; transition:color .15s ease; }}
    .nav-links a:hover {{ color:var(--teal-dark); }}
    .nav-status {{ display:flex; justify-content:flex-end; align-items:center; gap:7px; }}
    .status-pill {{ display:inline-flex; align-items:center; gap:7px; color:var(--slate); border:1px solid var(--border); border-radius:999px; padding:6px 9px; font-size:.67rem; font-weight:600; white-space:nowrap; background:#fff; }}
    .status-dot {{ width:6px; height:6px; border-radius:50%; background:var(--teal); box-shadow:0 0 0 3px #CCFBF1; }}
    .dev-pill {{ color:#92400E; border-color:#FCD34D; background:#FFFBEB; }}

    /* Hero */
    .hero {{ padding:76px 0 56px; max-width:690px; }}
    .eyebrow {{ color:var(--teal-dark); font-size:.7rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }}
    .hero h1 {{ max-width:16ch; color:var(--navy); font-size:2.5rem; line-height:1.12; letter-spacing:-.02em; font-weight:700; margin:12px 0 16px; }}
    .hero p {{ max-width:650px; color:var(--muted); font-size:.98rem; line-height:1.7; margin:0; }}
    .section-anchor {{ scroll-margin-top:84px; }}

    /* Stepper and analysis cards */
    .stepper {{ display:flex; align-items:center; width:100%; margin:0 0 20px; }}
    .step {{ display:flex; align-items:center; gap:8px; color:#94A3B8; font-size:.76rem; font-weight:600; white-space:nowrap; }}
    .step.active {{ color:var(--teal-dark); font-weight:700; }}
    .step.complete {{ color:var(--slate); }}
    .step-number {{ display:grid; place-items:center; width:25px; height:25px; color:#94A3B8; border:1px solid #CBD5E1; border-radius:50%; font-size:.7rem; }}
    .step.active .step-number {{ color:#fff; background:var(--teal); border-color:var(--teal); box-shadow:0 0 0 4px #CCFBF1; }}
    .step.complete .step-number {{ color:#fff; background:var(--teal); border-color:var(--teal); }}
    .step-line {{ flex:1; height:1px; max-width:130px; margin:0 12px; background:#CBD5E1; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ height:100%; min-height:420px; background:#fff; border:1px solid var(--border); border-radius:14px; box-shadow:0 1px 3px rgba(15,23,42,.06), 0 8px 24px rgba(15,23,42,.05); }}
    [data-testid="stVerticalBlockBorderWrapper"] > div {{ padding:24px 26px; }}
    .card-heading {{ color:var(--navy); font-size:.98rem; font-weight:700; margin-bottom:5px; }}
    .card-subtitle {{ color:var(--muted); font-size:.75rem; line-height:1.5; margin-bottom:18px; }}

    /* File uploader */
    [data-testid="stFileUploader"] {{ width:100%; }}
    [data-testid="stFileUploaderDropzone"] {{ display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:205px; padding:18px; background:#fff; border:2px dashed #94A3B8; border-radius:10px; transition:border-color .15s ease, background .15s ease; }}
    [data-testid="stFileUploaderDropzone"]:hover {{ border-color:var(--teal); background:#F0FDFA; }}
    [data-testid="stFileUploaderDropzone"] > div {{ width:100%; }}
    [data-testid="stFileUploaderDropzoneInstructions"] {{ color:var(--muted); text-align:center; }}
    [data-testid="stFileUploaderDropzoneInstructions"] > div {{ justify-content:center; }}
    [data-testid="stFileUploaderDropzoneInstructions"] svg {{ color:var(--teal); }}
    [data-testid="stFileUploaderDropzoneInstructions"] span {{ color:var(--slate); font-size:.8rem; font-weight:600; }}
    [data-testid="stFileUploaderDropzoneInstructions"] small {{ color:var(--muted); font-size:.68rem; }}
    [data-testid="stFileUploaderDropzone"] button {{ color:#fff !important; background:var(--teal) !important; border:1px solid var(--teal) !important; border-radius:8px !important; font-weight:700 !important; box-shadow:none !important; }}
    [data-testid="stFileUploaderDropzone"] button:hover {{ color:#fff !important; background:var(--teal-dark) !important; border-color:var(--teal-dark) !important; }}
    [data-testid="stFileUploaderFile"] {{ border-color:var(--border); }}
    .upload-note {{ display:flex; align-items:center; gap:7px; color:var(--muted); font-size:.7rem; margin-top:15px; }}

    /* Result state */
    .empty-state {{ display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:320px; text-align:center; color:var(--muted); }}
    .empty-icon {{ display:grid; place-items:center; width:54px; height:54px; margin-bottom:15px; color:#94A3B8; background:#F1F5F9; border-radius:50%; }}
    .empty-state strong {{ color:var(--slate); font-size:.9rem; }}
    .empty-state p {{ max-width:270px; margin:6px 0 0; font-size:.76rem; line-height:1.55; }}
    .scan-preview {{ max-height:168px; object-fit:contain; border:1px solid var(--border); border-radius:8px; background:#F8FAFC; }}
    .result-header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin:8px 0 14px; }}
    .result-label {{ color:var(--muted); font-size:.67rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }}
    .result-class {{ color:var(--navy); font-size:1.15rem; font-weight:700; line-height:1.25; margin-top:4px; }}
    .severity {{ color:#fff; border-radius:999px; padding:5px 8px; font-size:.65rem; font-weight:700; white-space:nowrap; }}
    .confidence-row {{ display:flex; align-items:center; gap:16px; padding:13px 0; border-top:1px solid var(--border); border-bottom:1px solid var(--border); }}
    .confidence-ring {{ display:grid; place-items:center; position:relative; flex:0 0 auto; width:76px; height:76px; border-radius:50%; background:conic-gradient(var(--ring-color) var(--confidence), #E2E8F0 0); }}
    .confidence-ring::after {{ content:""; position:absolute; inset:6px; background:#fff; border-radius:50%; }}
    .confidence-ring span {{ position:relative; z-index:1; color:var(--navy); font-size:.9rem; font-weight:700; }}
    .confidence-copy {{ color:var(--muted); font-size:.7rem; line-height:1.5; }}
    .confidence-copy strong {{ display:block; color:var(--navy); font-size:.8rem; margin-bottom:2px; }}
    .probability-title {{ color:var(--navy); font-size:.74rem; font-weight:700; margin:14px 0 8px; }}
    .probability {{ margin-bottom:7px; }}
    .probability-label {{ display:flex; justify-content:space-between; color:var(--slate); font-size:.67rem; margin-bottom:3px; }}
    .bar {{ height:5px; overflow:hidden; background:#E2E8F0; border-radius:999px; }}
    .bar-fill {{ height:100%; border-radius:999px; }}
    .info-box {{ margin-top:12px; padding:10px 12px; color:var(--slate); background:#F8FAFC; border:1px solid var(--border); border-radius:8px; font-size:.7rem; line-height:1.5; }}
    .info-box strong {{ color:var(--navy); }}
    .low-confidence {{ margin-top:10px; padding:8px 10px; color:#92400E; background:#FFFBEB; border-left:3px solid var(--amber); border-radius:5px; font-size:.7rem; }}
    .error-card {{ padding:14px; color:#991B1B; background:#FEF2F2; border:1px solid #FECACA; border-left:3px solid #DC2626; border-radius:8px; font-size:.78rem; line-height:1.5; }}

    /* Full-width disclaimer and content sections */
    .disclaimer {{ margin:22px 0 60px; padding:13px 16px; color:#78350F; background:#FFFBEB; border:1px solid #FDE68A; border-left:4px solid var(--amber); border-radius:8px; font-size:.76rem; line-height:1.55; }}
    .section {{ scroll-margin-top:84px; padding:58px 0 0; }}
    .section-kicker {{ color:var(--teal-dark); font-size:.68rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }}
    .section-title {{ color:var(--navy); font-size:1.55rem; letter-spacing:-.02em; font-weight:700; margin:8px 0 8px; }}
    .section-copy {{ max-width:620px; color:var(--muted); font-size:.82rem; line-height:1.6; margin:0 0 22px; }}
    .feature-card {{ height:100%; min-height:145px; padding:20px; background:#fff; border:1px solid var(--border); border-radius:12px; box-shadow:0 1px 3px rgba(15,23,42,.04); }}
    .feature-icon {{ display:grid; place-items:center; width:32px; height:32px; margin-bottom:15px; color:var(--teal); background:#F0FDFA; border-radius:8px; }}
    .feature-card h3 {{ color:var(--navy); font-size:.83rem; margin:0 0 7px; }}
    .feature-card p {{ color:var(--muted); font-size:.72rem; line-height:1.5; margin:0; }}
    .model-card {{ padding:8px 22px; background:#fff; border:1px solid var(--border); border-radius:12px; box-shadow:0 1px 3px rgba(15,23,42,.04); }}
    .spec-row {{ display:grid; grid-template-columns:170px 1fr; gap:20px; padding:14px 0; border-bottom:1px solid var(--border); font-size:.76rem; }}
    .spec-row:last-child {{ border-bottom:0; }}
    .spec-label {{ color:var(--muted); font-weight:600; }}
    .spec-value {{ color:var(--slate); font-weight:500; }}
    .kpi {{ height:100%; padding:19px; background:#fff; border:1px solid var(--border); border-radius:12px; box-shadow:0 1px 3px rgba(15,23,42,.04); }}
    .kpi-value {{ color:var(--navy); font-size:1.55rem; font-weight:700; letter-spacing:-.03em; }}
    .kpi-label {{ color:var(--muted); font-size:.7rem; margin-top:5px; }}
    .performance-note {{ color:var(--muted); font-size:.75rem; margin:18px 0 0; }}
    .roadmap {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:10px 28px; max-width:760px; padding:20px; background:#fff; border:1px solid var(--border); border-radius:12px; }}
    .roadmap-item {{ display:flex; align-items:flex-start; gap:9px; color:var(--slate); font-size:.76rem; line-height:1.5; }}
    .roadmap-item svg {{ flex:0 0 auto; margin-top:1px; color:var(--teal); }}
    .footer {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-top:76px; padding:22px 0; color:#94A3B8; border-top:1px solid var(--border); font-size:.7rem; }}
    .footer a {{ color:var(--slate); font-weight:600; text-decoration:none; }}

    @media (max-width:900px) {{
        .nav-links {{ display:none; }} .navbar {{ display:flex; justify-content:space-between; }}
        .hero {{ padding-top:56px; }} .roadmap {{ grid-template-columns:1fr; }}
    }}
    @media (max-width:620px) {{
        .block-container {{ padding:0 1rem 1rem; }} .navbar {{ margin:0 -1rem; padding:0 1rem; }}
        .dev-pill {{ display:none; }} .hero h1 {{ font-size:2.2rem; }}
        .step {{ font-size:.66rem; }} .step-line {{ margin:0 6px; }}
        [data-testid="stVerticalBlockBorderWrapper"] > div {{ padding:20px 18px; }}
        .spec-row {{ grid-template-columns:1fr; gap:4px; }} .footer {{ flex-direction:column; align-items:flex-start; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model loading: unchanged inference contract and custom focal loss.
# -----------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, name="categorical_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        ce = -y_true * tf.math.log(y_pred)
        return tf.reduce_mean(tf.reduce_sum(focal_weight * ce, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma})
        return config


@st.cache_resource
def load_model():
    if MODEL_PATH is None:
        raise FileNotFoundError("Model file not found. Place best_model_augmented_focal.keras in the project folder.")
    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"CategoricalFocalLoss": CategoricalFocalLoss},
        )
    except ValueError:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)


# -----------------------------------------------------------------------------
# Small reusable UI renderers.
# -----------------------------------------------------------------------------
def render_stepper(active_step):
    labels = ["Upload OCT Scan", "AI Analysis", "Review Result"]
    parts = ['<div class="stepper">']
    for index, label in enumerate(labels, start=1):
        state = "active" if index == active_step else "complete" if index < active_step else ""
        number = svg_icon("check", 14, "#FFFFFF") if index < active_step else str(index)
        parts.append(f'<div class="step {state}"><span class="step-number">{number}</span><span>{label}</span></div>')
        if index < len(labels):
            parts.append('<div class="step-line"></div>')
    parts.append("</div>")
    return "".join(parts)


def render_error(message):
    safe_message = html.escape(message)
    st.markdown(f'<div class="error-card"><strong>Analysis unavailable</strong><br>{safe_message}</div>', unsafe_allow_html=True)


def render_result(predicted_class, confidence, predictions, image):
    meta = CLASS_META[predicted_class]
    info = CLASS_INFO[predicted_class]
    st.image(image, caption="Uploaded OCT scan", use_container_width=True)
    st.markdown(
        f"""
        <div class="result-header">
            <div><div class="result-label">Predicted category</div><div class="result-class">{info['title']}</div></div>
            <span class="severity" style="background:{meta['color']};">{predicted_class} · {meta['severity']}</span>
        </div>
        <div class="confidence-row">
            <div class="confidence-ring" style="--confidence:{confidence:.1f}%;--ring-color:{meta['color']};"><span>{confidence:.1f}%</span></div>
            <div class="confidence-copy"><strong>Model confidence</strong>Highest-probability classification for this scan.</div>
        </div>
        <div class="probability-title">Class probability breakdown</div>
        """,
        unsafe_allow_html=True,
    )
    sorted_predictions = sorted(zip(CLASS_NAMES, predictions * 100), key=lambda item: item[1], reverse=True)
    for class_name, probability in sorted_predictions:
        color = CLASS_META[class_name]["color"]
        weight = "700" if class_name == predicted_class else "500"
        st.markdown(
            f'<div class="probability"><div class="probability-label"><span style="font-weight:{weight};">{class_name}</span><span>{probability:.1f}%</span></div><div class="bar"><div class="bar-fill" style="width:{probability:.1f}%;background:{color};"></div></div></div>',
            unsafe_allow_html=True,
        )
    if confidence < 70:
        st.markdown('<div class="low-confidence">Low confidence - interpret with caution.</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="info-box"><strong>About this condition</strong><br>{info["description"]}<br><br><strong>Suggested next step</strong><br>{info["recommendation"]}</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Navbar and hero.
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <nav class="navbar">
        <a class="brand" href="#top"><img src="{logo_data_uri()}" alt="RetinaAI logo"><span>RetinaAI</span></a>
        <div class="nav-links">
            <a href="#analyze">Analyze</a><a href="#about">About</a><a href="#model">Model</a>
            <a href="#performance">Performance</a><a href="{DOCUMENTATION_URL}" target="_blank" rel="noopener">Documentation</a>
            <a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub</a>
        </div>
        <div class="nav-status"><span class="status-pill"><span class="status-dot"></span>AI-Powered</span><span class="status-pill dev-pill">Under Development</span></div>
    </nav>
    <div id="top" class="hero">
        <div class="eyebrow">Clinical imaging workspace</div>
        <h1>Retinal OCT analysis, made clear.</h1>
        <p>Upload an Optical Coherence Tomography scan to receive an AI-assisted screening across four categories: CNV, DME, Drusen, and Normal.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Analyze: dynamic stepper, upload card, and result card.
# -----------------------------------------------------------------------------
stepper_placeholder = st.empty()
stepper_placeholder.markdown(render_stepper(1), unsafe_allow_html=True)

analyze_anchor = st.empty()
st.markdown('<div id="analyze" class="section-anchor"></div>', unsafe_allow_html=True)
col_left, col_right = st.columns([0.95, 1.05], gap="large")

uploaded_file = None
image = None
invalid_upload = None
predictions = None
predicted_class = None
confidence = None

with col_left:
    with st.container(border=True):
        st.markdown('<div class="card-heading">Upload OCT scan</div><div class="card-subtitle">JPG, JPEG, or PNG files · up to 200 MB</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop your OCT scan here",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )
        st.markdown(f'<div class="upload-note">{svg_icon("eye", 15, "#0D9488")} Secure local preview · Research use only</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        invalid_upload = str(error)

with col_right:
    with st.container(border=True):
        st.markdown('<div class="card-heading">Analysis result</div><div class="card-subtitle">AI-assisted classification report</div>', unsafe_allow_html=True)
        if uploaded_file is None:
            st.markdown(
                f'<div class="empty-state"><div class="empty-icon">{svg_icon("eye", 26, "#94A3B8")}</div><strong>No scan uploaded yet</strong><p>Upload an OCT image to generate a diagnostic report.</p></div>',
                unsafe_allow_html=True,
            )
        elif invalid_upload:
            render_error("The uploaded file could not be read as a valid image.")
        else:
            stepper_placeholder.markdown(render_stepper(2), unsafe_allow_html=True)
            try:
                with st.spinner("Running AI analysis..."):
                    # Preserve the original 224x224 RGB normalization and class order.
                    image_array = np.array(image.resize((224, 224))) / 255.0
                    predictions = load_model().predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
                    predicted_index = int(np.argmax(predictions))
                    predicted_class = CLASS_NAMES[predicted_index]
                    confidence = float(predictions[predicted_index]) * 100
                render_result(predicted_class, confidence, predictions, image)
                stepper_placeholder.markdown(render_stepper(3), unsafe_allow_html=True)
            except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
                render_error(str(error))


# -----------------------------------------------------------------------------
# Full-width disclaimer.
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="disclaimer"><strong>Clinical disclaimer</strong><br>This tool is for educational and research purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified ophthalmologist.</div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# About, model, performance, and roadmap sections.
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <section id="about" class="section">
        <div class="section-kicker">Why this model</div><div class="section-title">Designed for practical retinal screening research</div>
        <p class="section-copy">RetinaAI combines transfer learning with techniques chosen to address the class imbalance visible in retinal OCT datasets.</p>
    </section>
    """,
    unsafe_allow_html=True,
)
feature_columns = st.columns(3, gap="large")
features = [
    ("brain", "Transfer learning", "MobileNetV3Large starts from ImageNet-pretrained visual features for efficient clinical image representation."),
    ("target", "Focal Loss", "The loss function gives more attention to harder and underrepresented examples during training."),
    ("spark", "Targeted augmentation", "Flip, rotation, and zoom augmentation improve exposure to varied retinal scan patterns."),
]
for column, (icon, title, copy) in zip(feature_columns, features):
    with column:
        st.markdown(f'<div class="feature-card"><div class="feature-icon">{svg_icon(icon, 17, "#0D9488")}</div><h3>{title}</h3><p>{copy}</p></div>', unsafe_allow_html=True)

st.markdown('<section id="model" class="section"><div class="section-kicker">Model specification</div><div class="section-title">A focused transfer-learning pipeline</div></section>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="model-card">
        <div class="spec-row"><div class="spec-label">Architecture</div><div class="spec-value">MobileNetV3Large</div></div>
        <div class="spec-row"><div class="spec-label">Pretraining</div><div class="spec-value">ImageNet pretrained</div></div>
        <div class="spec-row"><div class="spec-label">Input</div><div class="spec-value">224 × 224 × 3 RGB OCT image</div></div>
        <div class="spec-row"><div class="spec-label">Loss</div><div class="spec-value">Categorical Focal Loss</div></div>
        <div class="spec-row"><div class="spec-label">Optimizer</div><div class="spec-value">Adam</div></div>
        <div class="spec-row"><div class="spec-label">Augmentation</div><div class="spec-value">Flip, rotation, and zoom</div></div>
        <div class="spec-row"><div class="spec-label">Framework</div><div class="spec-value">TensorFlow / Keras · Streamlit Cloud</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<section id="performance" class="section"><div class="section-kicker">Evaluation snapshot</div><div class="section-title">Measured performance</div><p class="section-copy">Reported metrics from the focal-loss model evaluation. These figures describe the research model, not a clinical diagnosis.</p></section>', unsafe_allow_html=True)
kpi_columns = st.columns(4, gap="medium")
kpis = [("94.76%", "Test accuracy"), ("0.91", "Macro F1"), ("93%", "DME recall"), ("75%", "DRUSEN recall")]
for column, (value, label) in zip(kpi_columns, kpis):
    with column:
        st.markdown(f'<div class="kpi"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)
st.markdown('<p class="performance-note">Drusen recall (75%) remains the main area for improvement.</p>', unsafe_allow_html=True)
if os.path.exists(CONFUSION_MATRIX_PATH):
    with st.expander("View confusion matrix"):
        st.image(CONFUSION_MATRIX_PATH, caption="Focal Loss model confusion matrix", use_container_width=True)

st.markdown('<section class="section"><div class="section-kicker">Roadmap</div><div class="section-title">Future work</div><p class="section-copy">Next steps for improving explainability, generalization, and deployment efficiency.</p></section>', unsafe_allow_html=True)
roadmap_items = ["Grad-CAM / Grad-CAM++ explainability", "Improve minority-class performance", "Lightweight deployment", "Multi-label classification", "External dataset evaluation"]
roadmap_html = ''.join(f'<div class="roadmap-item">{svg_icon("check", 15, "#0D9488")}<span>{item}</span></div>' for item in roadmap_items)
st.markdown(f'<div class="roadmap">{roadmap_html}</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Light footer.
# -----------------------------------------------------------------------------
st.markdown(
    f'<footer class="footer"><span>RetinaAI - Research &amp; educational use only</span><span><a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub</a> · ridoy-adhikary</span></footer>',
    unsafe_allow_html=True,
)
