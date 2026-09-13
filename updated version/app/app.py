import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime

# ------------------------------------------------------------------
# 0. GLOBALS
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "..", "best_model_augmented_focal.keras"),
    os.path.join(BASE_DIR, "best_model_augmented_focal.keras"),
    os.path.join(BASE_DIR, "..", "final_eye_disease_model.keras"),
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)

# ------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------------
st.set_page_config(
    page_title="RetinaAI | OCT Diagnostic Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# 2. DESIGN SYSTEM / THEME
# ------------------------------------------------------------------
_PRIMARY = "#0b3d91"
_PRIMARY_SOFT = "#1e88e5"
_TEAL = "#00acc1"
_BG = "#b9c4d3"
_CARD = "#FFD0D0"
_TEXT = "#1a2433"
_MUTED = "#64748b"
_BORDER = "#171a1f"

FONT_FAMILY = "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"

# Category accent + severity mapping for the four classes
CLASS_META = {
    "CNV":     {"color": "#e53935", "dot": "🔴", "severity": "High Priority"},
    "DME":     {"color": "#fb8c00", "dot": "🟠", "severity": "Moderate Priority"},
    "DRUSEN":  {"color": "#f9a825", "dot": "🟡", "severity": "Monitor"},
    "NORMAL":  {"color": "#43a047", "dot": "🟢", "severity": "Healthy"},
}

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: {FONT_FAMILY};
    }}

    /* ---- Global background ---- */
    .stApp {{
        background-color: {_BG};
    }}

    /* ---- Hide default Streamlit chrome for a cleaner look ---- */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    [data-testid="stToolbar"] {{visibility: hidden;}}

    /* ---- Top navigation bar ---- */
    .topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(90deg, #0b3d91 0%, #1558b0 45%, #00acc1 100%);
        border-radius: 14px;
        padding: 0.9rem 1.6rem;
        margin-bottom: 1.6rem;
        color: #ffffff;
        box-shadow: 0 6px 18px rgba(11, 61, 145, 0.25);
    }}
    .topbar-brand {{
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 0.2px;
    }}
    .topbar-brand img {{
        width: 42px;
        height: 42px;
    }}
    .topbar-tag {{
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.35);
        padding: 0.3rem 0.9rem;
        border-radius: 40px;
        font-size: 0.8rem;
        font-weight: 600;
    }}

    /* ---- Hero ---- */
    .hero {{
        text-align: center;
        padding: 0.2rem 0 0.4rem 0;
    }}
    .hero h1 {{
        font-size: 2.5rem;
        font-weight: 800;
        color: {_PRIMARY};
        margin: 0;
        line-height: 1.15;
    }}
    .hero .sub {{
        color: {_MUTED};
        font-size: 1.05rem;
        margin-top: 0.4rem;
    }}
    .pill-row {{
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }}
    .pill {{
        background: #ffffff;
        border: 1px solid {_BORDER};
        border-radius: 40px;
        padding: 0.35rem 1rem;
        font-size: 0.82rem;
        font-weight: 600;
        color: {_PRIMARY};
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
    }}

    /* ---- Section titles ---- */
    .section-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {_TEXT};
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.9rem;
    }}
    .section-title .num {{
        width: 26px;
        height: 26px;
        background: {_PRIMARY};
        color: #fff;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
    }}

    /* ---- Upload zone ---- */
    [data-testid="stFileUploaderDropzone"] {{
        background: #ffffff;
        border: 2px dashed #b9c8e6;
        border-radius: 14px;
        padding: 1.4rem;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {_PRIMARY_SOFT};
        background: #f6faff;
    }}

    /* ---- Cards ---- */
    .card {{
        background: {_CARD};
        border: 1px solid {_BORDER};
        border-radius: 16px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
        margin-bottom: 1.1rem;
    }}
    .card-upload {{min-height: 220px;}}

    /* ---- Result card ---- */
    .result-card {{
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid {_BORDER};
        border-left: 8px solid {_PRIMARY_SOFT};
        padding: 1.5rem 1.6rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.1rem;
    }}
    .result-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.6rem;
    }}
    .result-class {{
        font-size: 1.9rem;
        font-weight: 800;
        color: {_TEXT};
    }}
    .badge {{
        font-size: 0.82rem;
        font-weight: 700;
        padding: 0.4rem 1rem;
        border-radius: 40px;
        color: #fff;
    }}
    .confidence {{
        font-size: 1.05rem;
        color: {_TEXT};
        margin-top: 0.35rem;
    }}
    .confidence b {{color: {_PRIMARY_SOFT};}}

    /* ---- Info box ---- */
    .info-box {{
        background: #f0f6ff;
        border: 1px solid #d7e6ff;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        font-size: 0.93rem;
        color: {_TEXT};
        margin-top: 1rem;
        line-height: 1.55;
    }}
    .info-box strong {{color: {_PRIMARY};}}

    /* ---- Disclaimer / footer ---- */
    .disclaimer {{
        background: #fff8e6;
        border: 1px solid #ffe9a8;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        font-size: 0.85rem;
        color: #7a5a00;
        line-height: 1.5;
    }}
    .footer {{
        text-align: center;
        color: {_MUTED};
        font-size: 0.85rem;
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid {_BORDER};
    }}

    /* ---- Buttons ---- */
    .stButton>button {{
        background: {_PRIMARY};
        color: #ffffff;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
        border: none;
        box-shadow: 0 3px 10px rgba(11, 61, 145, 0.2);
    }}
    .stButton>button:hover {{
        background: {_PRIMARY_SOFT};
        color: #ffffff;
        border: none;
    }}

    /* ---- Misc ---- */
    .stProgress > div > div > div > div {{
        background: {_PRIMARY_SOFT};
    }}
    [data-testid="stSidebar"] {{
        background: #ffffff;
        border-right: 1px solid {_BORDER};
    }}
    [data-testid="stSidebar"] .block-container {{padding-top: 1.4rem;}}
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. MODEL LOADING (CACHED)
# ------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, name="categorical_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1. - tf.keras.backend.epsilon())
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1. - p_t, self.gamma)
        ce = -y_true * tf.math.log(y_pred)
        return tf.reduce_mean(tf.reduce_sum(focal_weight * ce, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@st.cache_resource
def load_model():
    if MODEL_PATH is None:
        st.error("❌ Model file not found. Place "
                 "'best_model_augmented_focal.keras' in the project folder.")
        st.stop()
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"CategoricalFocalLoss": CategoricalFocalLoss}
        )
    except ValueError:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

# ------------------------------------------------------------------
# 4. METADATA
# ------------------------------------------------------------------
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_INFO = {
    "CNV": {
        "title": "Choroidal Neovascularization (CNV)",
        "description": "Abnormal blood vessel growth beneath the retina. A leading cause "
                        "of severe, rapid vision loss in age-related macular degeneration (AMD).",
        "recommendation": "Consult a retina specialist immediately for urgent evaluation.",
    },
    "DME": {
        "title": "Diabetic Macular Edema (DME)",
        "description": "Fluid accumulation in the macula from leaking retinal vessels, "
                        "common in diabetic retinopathy.",
        "recommendation": "Schedule a prompt ophthalmology appointment; blood-sugar control is key.",
    },
    "DRUSEN": {
        "title": "Drusen",
        "description": "Yellow deposits beneath the retina, often an early indicator of "
                        "age-related macular degeneration (AMD).",
        "recommendation": "Monitor with regular eye exams and track progression over time.",
    },
    "NORMAL": {
        "title": "Normal Retina",
        "description": "No signs of CNV, DME, or Drusen detected. The retinal layers appear "
                        "healthy and well-defined.",
        "recommendation": "Excellent — continue routine eye checkups.",
    },
}

# ------------------------------------------------------------------
# 5. TOP NAVIGATION BAR
# ------------------------------------------------------------------
st.markdown(f"""
    <div class="topbar">
        <div class="topbar-brand">
            <img src="https://img.icons8.com/fluency/96/000000/eye.png" alt="logo"/>
            <span>RetinaAI</span>
        </div>
        <div class="topbar-tag">AI-Powered OCT Diagnostic Assistant</div>
    </div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 6. HERO
# ------------------------------------------------------------------
st.markdown("""
    <div class="hero">
        <h1>Retinal OCT Diagnostic Assistant</h1>
        <div class="sub">Upload an Optical Coherence Tomography (OCT) scan and receive an instant,
        AI-assisted screening for four common retinal conditions.</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="pill-row">
        <span class="pill">🧠 MobileNetV3Large</span>
        <span class="pill">🎯 Focal Loss + Augmentation</span>
        <span class="pill">📈 ~95% Test Accuracy</span>
        <span class="pill">🩺 4 Retinal Classes</span>
    </div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 7. PROCESS STEPS + MAIN LAYOUT
# ------------------------------------------------------------------
steps = st.columns(3)
step_labels = ["Upload OCT Scan", "AI Analysis", "Review Result"]
for i, (col, label) in enumerate(zip(steps, step_labels)):
    with col:
        st.markdown(
            f'<div class="card" style="text-align:center;padding:0.7rem;">'
            f'<span class="num" '
            f'style="background:{_PRIMARY if i <= 0 else "#cbd5e1"};color:'
            f'{"#fff" if i <= 0 else "#334155"};">{i+1}</span>&nbsp;'
            f'<span style="font-weight:600;color:{_TEXT};">{label}</span></div>',
            unsafe_allow_html=True,
        )

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown(
        '<div class="section-title"><span class="num">📤</span> Upload OCT Image</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Choose an OCT scan (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded OCT Scan", use_container_width=True)

    st.markdown('<div class="disclaimer">⚠️ &nbsp;This tool is for <b>educational and research '
                'purposes only</b>. It is not a substitute for professional medical diagnosis. '
                'Always consult a qualified ophthalmologist.</div>', unsafe_allow_html=True)

with col_right:
    st.markdown(
        '<div class="section-title"><span class="num">🔬</span> Analysis Result</div>',
        unsafe_allow_html=True,
    )

    if uploaded_file is None:
        st.markdown("""
            <div class="card card-upload">
                <div style="text-align:center;padding-top:2.2rem;color:#94a3b8;">
                    <div style="font-size:3rem;">👁️</div>
                    <div style="font-weight:600;margin-top:0.5rem;">Awaiting input</div>
                    <div style="font-size:0.9rem;margin-top:0.3rem;">
                        Upload an OCT scan on the left to begin the assessment.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Running deep-learning analysis..."):
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            model = load_model()
            predictions = model.predict(img_array, verbose=0)[0]

            idx = int(np.argmax(predictions))
            pred_class = CLASS_NAMES[idx]
            confidence = float(predictions[idx]) * 100

        meta = CLASS_META[pred_class]
        info = CLASS_INFO[pred_class]

        # --- Result card ---
        st.markdown(f"""
            <div class="result-card" style="border-left-color:{meta['color']};">
                <div class="result-top">
                    <span class="result-class">{meta['dot']}&nbsp;{info['title']}</span>
                    <span class="badge" style="background:{meta['color']};">
                        {meta['severity']}
                    </span>
                </div>
                <div class="confidence">Model confidence: <b>{confidence:.1f}%</b></div>
                <div class="info-box">
                    <strong>{info['description']}</strong><br><br>
                    <strong>Recommendation:</strong> {info['recommendation']}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- Per-class confidence meter ---
        st.markdown('#### 📊 Class Probability Breakdown')
        for cls, prob in zip(CLASS_NAMES, predictions * 100):
            c = CLASS_META[cls]["color"]
            highlight = cls == pred_class
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.9rem;color:{_TEXT};margin-bottom:2px;">'
                f'<span style="font-weight:{"700" if highlight else "500"};'
                f'color:{"#000" if highlight else _MUTED};">{cls}</span>'
                f'<span style="font-weight:{"700" if highlight else "500"};">{prob:.1f}%</span>'
                f'</div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div style="width:100%;background:#eef2f7;border-radius:6px;height:8px;margin-bottom:0.55rem;">
                    <div style="width:{prob:.1f}%;background:{c};height:8px;border-radius:6px;"></div>
                </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 8. FOOTER
# ------------------------------------------------------------------
st.markdown(
    f'<div class="footer">Built with ❤️ · Streamlit · TensorFlow · MobileNetV3Large · '
    f'© {datetime.now().year} RetinaAI</div>',
    unsafe_allow_html=True,
)