import base64
import os
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "..", "best_model_augmented_focal.keras"),
    os.path.join(BASE_DIR, "best_model_augmented_focal.keras"),
    os.path.join(BASE_DIR, "..", "final_eye_disease_model.keras"),
]
MODEL_PATH = next((path for path in MODEL_CANDIDATES if os.path.exists(path)), None)

st.set_page_config(
    page_title="RetinaAI | OCT Diagnostic Assistant",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

NAVY = "#0F172A"
SLATE = "#334155"
TEAL = "#0D9488"
TEAL_DARK = "#0F766E"
BG = "#F8FAFC"
MUTED = "#64748B"
BORDER = "#E2E8F0"

CLASS_META = {
    "CNV": {"color": "#DC2626", "severity": "High priority"},
    "DME": {"color": "#D97706", "severity": "Moderate priority"},
    "DRUSEN": {"color": "#B45309", "severity": "Monitor"},
    "NORMAL": {"color": "#059669", "severity": "Healthy"},
}

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_INFO = {
    "CNV": {
        "title": "Choroidal Neovascularization (CNV)",
        "description": "Abnormal blood vessel growth beneath the retina. A leading cause of severe, rapid vision loss in age-related macular degeneration (AMD).",
        "recommendation": "Consult a retina specialist immediately for urgent evaluation.",
    },
    "DME": {
        "title": "Diabetic Macular Edema (DME)",
        "description": "Fluid accumulation in the macula from leaking retinal vessels, common in diabetic retinopathy.",
        "recommendation": "Schedule a prompt ophthalmology appointment; blood-sugar control is key.",
    },
    "DRUSEN": {
        "title": "Drusen",
        "description": "Yellow deposits beneath the retina, often an early indicator of age-related macular degeneration (AMD).",
        "recommendation": "Monitor with regular eye exams and track progression over time.",
    },
    "NORMAL": {
        "title": "Normal Retina",
        "description": "No signs of CNV, DME, or Drusen detected. The retinal layers appear healthy and well-defined.",
        "recommendation": "Continue routine eye checkups.",
    },
}


def logo_data_uri():
    logo_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
    <rect width="64" height="64" rx="16" fill="#0D9488"/>
    <path d="M10 32s8-14 22-14 22 14 22 14-8 14-22 14S10 32 10 32Z" fill="none" stroke="#fff" stroke-width="4"/>
    <circle cx="32" cy="32" r="7" fill="#fff"/>
    <circle cx="32" cy="32" r="3" fill="#0D9488"/>
    </svg>"""
    encoded = base64.b64encode(logo_svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


st.markdown(
    f"""
    <style>
    :root {{
        --navy: {NAVY}; --slate: {SLATE}; --teal: {TEAL};
        --teal-dark: {TEAL_DARK}; --bg: {BG}; --muted: {MUTED};
        --border: {BORDER};
    }}
    html, body, [class*="css"] {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .stApp {{ background: var(--bg); color: var(--navy); }}
    .block-container {{ max-width: 1240px; padding: 2rem 2.5rem 1rem; }}
    #MainMenu, [data-testid="stToolbar"], footer {{ visibility: hidden; }}
    [data-testid="stSidebar"] {{ display: none; }}

    .topbar {{ display:flex; align-items:center; justify-content:space-between; padding:0.25rem 0 2.3rem; border-bottom:1px solid var(--border); }}
    .brand {{ display:flex; align-items:center; gap:0.7rem; color:var(--navy); font-size:1.25rem; font-weight:750; letter-spacing:-0.02em; }}
    .brand img {{ width:38px; height:38px; border-radius:11px; }}
    .tag {{ display:inline-flex; align-items:center; gap:0.45rem; color:var(--slate); background:#fff; border:1px solid var(--border); border-radius:999px; padding:0.45rem 0.8rem; font-size:0.75rem; font-weight:650; }}
    .tag-dot {{ width:7px; height:7px; border-radius:50%; background:var(--teal); box-shadow:0 0 0 4px #CCFBF1; }}
    .intro {{ padding:2.35rem 0 1.5rem; }}
    .eyebrow {{ color:var(--teal-dark); font-size:0.72rem; font-weight:750; letter-spacing:0.11em; text-transform:uppercase; }}
    h1 {{ color:var(--navy); font-size:clamp(2rem, 4vw, 3.2rem); line-height:1.06; letter-spacing:-0.055em; margin:0.5rem 0 0.8rem; }}
    .intro-copy {{ max-width:650px; color:var(--muted); font-size:1rem; line-height:1.65; margin:0; }}
    .stepper {{ display:flex; align-items:center; margin:0 0 2rem; }}
    .step {{ display:flex; align-items:center; gap:0.55rem; color:#94A3B8; font-size:0.82rem; font-weight:600; white-space:nowrap; }}
    .step.active {{ color:var(--teal-dark); font-weight:750; }}
    .step-number {{ display:grid; place-items:center; width:25px; height:25px; border:1px solid #CBD5E1; border-radius:50%; font-size:0.72rem; }}
    .step.active .step-number {{ color:white; background:var(--teal); border-color:var(--teal); box-shadow:0 0 0 4px #CCFBF1; }}
    .step-line {{ height:1px; flex:1; max-width:100px; background:#CBD5E1; margin:0 0.75rem; }}
    .panel {{ background:#fff; border:1px solid var(--border); border-radius:12px; padding:1.35rem; box-shadow:0 8px 24px rgba(15,23,42,0.04); height:100%; box-sizing:border-box; }}
    .panel-title {{ color:var(--navy); font-size:1rem; font-weight:750; margin:0 0 0.25rem; }}
    .panel-subtitle {{ color:var(--muted); font-size:0.82rem; margin:0 0 1rem; }}
    [data-testid="stFileUploaderDropzone"] {{ background:#F8FAFC; border:1px dashed #94A3B8; border-radius:10px; min-height:190px; padding:1rem; }}
    [data-testid="stFileUploaderDropzone"]:hover {{ border-color:var(--teal); background:#F0FDFA; }}
    [data-testid="stFileUploaderDropzone"] small {{ color:var(--muted); }}
    .disclaimer {{ margin-top:1rem; padding:0.85rem 1rem; color:#92400E; background:#FFFBEB; border:1px solid #FDE68A; border-left:3px solid #F59E0B; border-radius:7px; font-size:0.78rem; line-height:1.55; }}
    .empty {{ min-height:340px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:1rem; color:var(--muted); }}
    .empty-icon {{ width:52px; height:52px; display:grid; place-items:center; margin-bottom:1rem; color:#94A3B8; background:#F1F5F9; border-radius:50%; }}
    .empty-icon svg {{ width:25px; height:25px; }}
    .empty strong {{ color:var(--slate); font-size:0.95rem; }}
    .empty p {{ max-width:300px; margin:0.45rem 0 0; font-size:0.82rem; line-height:1.55; }}
    .scan-preview {{ max-height:220px; width:100%; object-fit:contain; border:1px solid var(--border); border-radius:8px; background:#F8FAFC; }}
    .result-header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; margin:1rem 0 1.1rem; }}
    .result-label {{ color:var(--muted); font-size:0.72rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; }}
    .result-class {{ color:var(--navy); font-size:1.45rem; font-weight:800; line-height:1.15; margin-top:0.3rem; }}
    .severity {{ color:#fff; border-radius:999px; padding:0.35rem 0.65rem; font-size:0.7rem; font-weight:700; white-space:nowrap; }}
    .confidence-row {{ display:flex; align-items:center; gap:1.2rem; padding:1rem 0; border-top:1px solid var(--border); border-bottom:1px solid var(--border); }}
    .confidence-ring {{ display:grid; place-items:center; width:88px; height:88px; border-radius:50%; flex:0 0 auto; background:conic-gradient(var(--ring-color) var(--confidence), #E2E8F0 0); position:relative; }}
    .confidence-ring::after {{ content:""; position:absolute; inset:7px; background:#fff; border-radius:50%; }}
    .confidence-ring span {{ position:relative; z-index:1; color:var(--navy); font-size:1.05rem; font-weight:800; }}
    .confidence-copy {{ color:var(--muted); font-size:0.78rem; line-height:1.5; }}
    .confidence-copy strong {{ display:block; color:var(--navy); font-size:0.92rem; }}
    .info-box {{ margin-top:1rem; padding:0.9rem 1rem; background:#F8FAFC; border:1px solid var(--border); border-radius:8px; color:var(--slate); font-size:0.8rem; line-height:1.55; }}
    .info-box strong {{ color:var(--navy); }}
    .probability-title {{ color:var(--navy); font-size:0.8rem; font-weight:750; margin:1.1rem 0 0.7rem; }}
    .probability {{ margin-bottom:0.55rem; }}
    .probability-label {{ display:flex; justify-content:space-between; color:var(--slate); font-size:0.72rem; margin-bottom:0.25rem; }}
    .bar {{ height:5px; background:#E2E8F0; border-radius:999px; overflow:hidden; }}
    .bar-fill {{ height:100%; border-radius:999px; }}
    .stButton > button {{ width:100%; color:#fff; background:var(--teal); border:1px solid var(--teal); border-radius:8px; font-weight:700; }}
    .stButton > button:hover {{ color:#fff; background:var(--teal-dark); border-color:var(--teal-dark); }}
    .footer {{ color:#94A3B8; text-align:center; font-size:0.72rem; padding:2.2rem 0 0.5rem; }}
    @media (max-width:700px) {{ .block-container {{ padding:1.25rem 1rem 0.75rem; }} .topbar {{ padding-bottom:1.5rem; }} .tag {{ font-size:0; padding:0.55rem; }} .tag-dot {{ margin:0; }} .step {{ font-size:0.68rem; }} .step-line {{ margin:0 0.35rem; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)


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
        st.error("Model file not found. Place 'best_model_augmented_focal.keras' in the project folder.")
        st.stop()
    try:
        return tf.keras.models.load_model(
            MODEL_PATH, custom_objects={"CategoricalFocalLoss": CategoricalFocalLoss}
        )
    except ValueError:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)


st.markdown(
    f"""
    <div class="topbar">
        <div class="brand"><img src="{logo_data_uri()}" alt="RetinaAI logo"><span>RetinaAI</span></div>
        <div class="tag"><span class="tag-dot"></span>AI-Powered OCT Diagnostic Assistant</div>
    </div>
    <div class="intro">
        <div class="eyebrow">Clinical imaging workspace</div>
        <h1>Retinal OCT analysis,<br>made clear.</h1>
        <p class="intro-copy">Upload an Optical Coherence Tomography scan to receive an AI-assisted screening across four common retinal conditions.</p>
    </div>
    <div class="stepper">
        <div class="step active"><span class="step-number">1</span><span>Upload OCT Scan</span></div>
        <div class="step-line"></div>
        <div class="step"><span class="step-number">2</span><span>AI Analysis</span></div>
        <div class="step-line"></div>
        <div class="step"><span class="step-number">3</span><span>Review Result</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([0.92, 1.08], gap="large")

with col_left:
    st.markdown(
        '<div class="panel"><div class="panel-title">Upload OCT scan</div><div class="panel-subtitle">JPG, JPEG, or PNG files up to 200 MB</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Choose an OCT scan",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded OCT scan", use_container_width=True)
    st.markdown(
        '<div class="disclaimer"><strong>Clinical disclaimer</strong><br>This tool is for educational and research purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified ophthalmologist.</div></div>',
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown('<div class="panel"><div class="panel-title">Analysis result</div><div class="panel-subtitle">AI-assisted classification report</div>', unsafe_allow_html=True)
    if uploaded_file is None:
        st.markdown(
            """
            <div class="empty">
                <div class="empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z"/><circle cx="12" cy="12" r="2.5"/></svg></div>
                <strong>No scan uploaded yet</strong>
                <p>Upload an OCT image to generate a diagnostic report.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Running AI analysis..."):
            image_array = np.array(image.resize((224, 224))) / 255.0
            predictions = load_model().predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
            predicted_index = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

        meta = CLASS_META[predicted_class]
        info = CLASS_INFO[predicted_class]
        st.markdown(
            f"""
            <div class="result-header"><div><div class="result-label">Predicted class</div><div class="result-class">{info['title']}</div></div><span class="severity" style="background:{meta['color']};">{meta['severity']}</span></div>
            <div class="confidence-row"><div class="confidence-ring" style="--confidence:{confidence:.1f}%;--ring-color:{meta['color']};"><span>{confidence:.1f}%</span></div><div class="confidence-copy"><strong>Model confidence</strong>The prediction reflects the model's highest-probability classification for this scan.</div></div>
            <div class="info-box"><strong>Clinical context</strong><br>{info['description']}<br><br><strong>Suggested next step</strong><br>{info['recommendation']}</div>
            <div class="probability-title">Class probability breakdown</div>
            """,
            unsafe_allow_html=True,
        )
        for class_name, probability in zip(CLASS_NAMES, predictions * 100):
            color = CLASS_META[class_name]["color"]
            weight = "700" if class_name == predicted_class else "500"
            st.markdown(
                f'<div class="probability"><div class="probability-label"><span style="font-weight:{weight};">{class_name}</span><span>{probability:.1f}%</span></div><div class="bar"><div class="bar-fill" style="width:{probability:.1f}%;background:{color};"></div></div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f'<div class="footer">RetinaAI · Streamlit · TensorFlow · © {datetime.now().year} RetinaAI</div>',
    unsafe_allow_html=True,
)
