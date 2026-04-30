"""
app.py
-------
CottonGuard AI — Cotton Plant Disease Detection & Treatment System
Streamlit UI with Image Upload, Live Webcam, GradCAM, and Treatment Recommendations.

Run:
    streamlit run app.py
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import cv2

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CottonGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a6b3c 0%, #2d9b5a 50%, #52c77c 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(26, 107, 60, 0.3);
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }
    .main-header p {
        color: rgba(255,255,255,0.85) !important;
        font-size: 1.05rem !important;
        margin: 0.3rem 0 0 0 !important;
    }

    /* Result cards */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #2d9b5a;
        margin-bottom: 1rem;
    }
    .result-card.danger { border-left-color: #e74c3c; }
    .result-card.warning { border-left-color: #f39c12; }
    .result-card.info { border-left-color: #3498db; }
    .result-card.success { border-left-color: #27ae60; }

    /* Disease name badge */
    .disease-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1a6b3c, #2d9b5a);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .healthy-badge {
        background: linear-gradient(135deg, #27ae60, #52c77c);
    }

    /* Confidence bar */
    .conf-bar-bg {
        background: #f0f2f5;
        border-radius: 50px;
        height: 14px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 50px;
        background: linear-gradient(90deg, #2d9b5a, #52c77c);
        transition: width 0.8s ease;
    }

    /* Medicine cards */
    .medicine-card {
        background: linear-gradient(135deg, #f8fffe, #f0fff8);
        border: 1px solid #c8e6c9;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .medicine-name {
        font-weight: 700;
        color: #1a6b3c;
        font-size: 1rem;
    }
    .medicine-detail {
        color: #555;
        font-size: 0.9rem;
        margin: 0.15rem 0;
    }

    /* Emergency alert */
    .emergency-box {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }
    .emergency-box.danger {
        background: linear-gradient(135deg, #fff5f5, #ffe8e8);
        border-color: #e74c3c;
    }

    /* Practice list */
    .practice-item {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-left: 3px solid #2d9b5a;
        font-size: 0.9rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d3d20 0%, #1a6b3c 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: rgba(255,255,255,0.8) !important; }

    /* Metrics */
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a6b3c;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Load Predictor (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor():
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from utils.predictor import Predictor
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        return Predictor("checkpoints/best.pth", device=device), None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


# ── Severity color mapping ─────────────────────────────────────────────────────
SEVERITY_COLOR = {
    "None": "success",
    "Medium": "info",
    "High": "warning",
    "Very High": "danger",
}

# ── Helper: Run prediction and display results ─────────────────────────────────
def run_prediction(predictor, image: Image.Image):
    from utils.treatments import get_treatment

    with st.spinner("🔬 Analyzing cotton leaf..."):
        result = predictor.predict(image)

    class_name = result["class_name"]
    confidence = result["confidence"]
    all_probs  = result["all_probs"]
    cam_image  = result["gradcam_overlay"]
    treatment  = get_treatment(class_name)

    severity    = treatment.get("severity", "None")
    card_style  = SEVERITY_COLOR.get(severity, "info")
    is_healthy  = class_name == "Healthy"

    # ── Row 1: Image + GradCAM ──────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("**📷 Uploaded Image**")
        st.image(image.resize((300, 300)), use_container_width=True)
    with col2:
        st.markdown("**🔥 GradCAM Attention Map**")
        st.image(cam_image, use_container_width=True)
        st.caption("Highlighted regions show where the model focused its attention")

    st.markdown("---")

    # ── Row 2: Diagnosis card ────────────────────────────────────────────────
    badge_class = "healthy-badge" if is_healthy else ""
    st.markdown(f"""
    <div class="result-card {card_style}">
        <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
            <span class="disease-badge {badge_class}">{class_name.replace('_', ' ')}</span>
            <span style="color:#888; font-size:0.9rem;">Severity: <b>{severity}</b></span>
        </div>
        <p style="margin:0.5rem 0 0; color:#555; font-size:0.9rem;">
            <b>Pathogen:</b> {treatment.get('pathogen', 'N/A')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence gauge ─────────────────────────────────────────────────────
    st.markdown(f"**Confidence Score: {confidence:.1f}%**")
    color = "#27ae60" if confidence > 70 else "#f39c12" if confidence > 50 else "#e74c3c"
    st.markdown(f"""
    <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{confidence}%; background: linear-gradient(90deg, {color}, {'#52c77c' if confidence > 70 else color});"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── All class probabilities ───────────────────────────────────────────────
    with st.expander("📊 All Class Probabilities"):
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs:
            bar_color = "#2d9b5a" if cls == class_name else "#94a3b8"
            st.markdown(f"""
            <div style="margin:0.3rem 0;">
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:2px;">
                    <span>{cls.replace('_', ' ')}</span><span><b>{prob:.1f}%</b></span>
                </div>
                <div class="conf-bar-bg">
                    <div style="height:100%; width:{prob}%; background:{bar_color}; border-radius:50px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Symptoms ─────────────────────────────────────────────────────────────
    if treatment.get("symptoms"):
        st.markdown("#### 🩺 Symptoms Observed")
        for s in treatment["symptoms"]:
            st.markdown(f'<div class="practice-item">• {s}</div>', unsafe_allow_html=True)

    # ── Emergency action ─────────────────────────────────────────────────────
    if not is_healthy:
        ea = treatment.get("emergency_action", "")
        urgency = "danger" if "URGENT" in ea else ""
        st.markdown(f"""
        <div class="emergency-box {urgency}">
            <b>{ea}</b>
        </div>
        """, unsafe_allow_html=True)

    # ── Treatment Medicines ───────────────────────────────────────────────────
    medicines = treatment.get("medicines", [])
    if medicines:
        st.markdown("#### 💊 Recommended Medicines & Treatments")
        for i, med in enumerate(medicines):
            st.markdown(f"""
            <div class="medicine-card">
                <div class="medicine-name">💊 {i+1}. {med['name']}</div>
                <div class="medicine-detail">📏 <b>Dose:</b> {med['dose']}</div>
                <div class="medicine-detail">🔁 <b>Frequency:</b> {med['frequency']}</div>
                <div class="medicine-detail">🌿 <b>Method:</b> {med['method']}</div>
                <div class="medicine-detail">🏷️ <b>Brand names:</b> {med['brand']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Your cotton plant appears healthy! No treatment needed.")

    # ── Cultural Practices ────────────────────────────────────────────────────
    if treatment.get("cultural_practices"):
        st.markdown("#### 🌾 Cultural Practices")
        for p in treatment["cultural_practices"]:
            st.markdown(f'<div class="practice-item">✔️ {p}</div>', unsafe_allow_html=True)

    # ── Prevention ────────────────────────────────────────────────────────────
    if treatment.get("prevention"):
        st.markdown(f"""
        <div class="result-card info" style="margin-top:1rem;">
            <b>🛡️ Prevention:</b> {treatment['prevention']}
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌿 CottonGuard AI</h1>
    <p>Cotton Plant Disease Detection & Treatment Recommendation System</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    mode = st.selectbox(
        "Detection Mode",
        ["📂 Upload Image", "📹 Live Webcam"],
        index=0,
    )
    st.markdown("---")
    st.markdown("### 🌱 Detectable Diseases")
    diseases = [
        ("🦠", "Bacterial Blight",     "High"),
        ("✅", "Healthy Cotton",        "None"),
        ("🍂", "Alternaria Leaf Spot",  "Medium"),
        ("🌀", "Curl Virus",           "Very High"),
        ("🥀", "Fusarium Wilt",        "High"),
    ]
    for icon, name, severity in diseases:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown(f"{icon} {name}")
        with col_b:
            st.markdown(f"`{severity}`")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **CottonGuard AI** uses EfficientNet-B0 with transfer learning to detect cotton plant diseases from leaf images.

    - 🔬 **GradCAM** visualizes what the model sees
    - 💊 **Treatment DB** with real medicines
    - 📹 **Live webcam** for field use
    """)

# ── Load model ─────────────────────────────────────────────────────────────────
predictor, error = load_predictor()

if error:
    st.error(f"""
    ⚠️ **Model not loaded:** {error}

    **To fix this, run training first:**
    ```bash
    python scripts/organize_dataset.py --src raw_images
    python train.py
    ```
    Then restart the app.
    """)
    st.stop()

st.success(f"✅ Model loaded | Device: {'GPU' if __import__('torch').cuda.is_available() else 'CPU'} | Classes: {len(predictor.classes)}")

# ── Stats row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-box"><div class="metric-value">5</div><div class="metric-label">Disease Classes</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-box"><div class="metric-value">B0</div><div class="metric-label">EfficientNet Model</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-box"><div class="metric-value">224px</div><div class="metric-label">Input Resolution</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-box"><div class="metric-value">GradCAM</div><div class="metric-label">Explainability</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# MODE: IMAGE UPLOAD
# ════════════════════════════════════════════════════════════════════════════════
if "Upload" in mode:
    st.markdown("## 📂 Upload Cotton Leaf Image")

    uploaded = st.file_uploader(
        "Drag & drop or click to upload a cotton leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear image of a cotton leaf for disease detection",
    )

    col_sample, _ = st.columns([2, 4])
    with col_sample:
        if st.button("🎲 Use Sample Image (for demo)"):
            # Generate a simple green leaf placeholder
            sample = Image.fromarray(
                np.random.randint(50, 200, (400, 400, 3), dtype=np.uint8)
            )
            st.session_state["sample_img"] = sample

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.markdown("---")
        st.markdown("## 🔬 Detection Results")
        run_prediction(predictor, image)

    elif "sample_img" in st.session_state:
        st.info("⚠️ Using a random placeholder image for demo. Upload a real cotton leaf for accurate results.")
        run_prediction(predictor, st.session_state["sample_img"])

    else:
        # Placeholder UI
        st.markdown("""
        <div style="border: 2px dashed #2d9b5a; border-radius: 12px; padding: 3rem;
                    text-align: center; background: #f8fff9; color: #888; margin-top: 1rem;">
            <div style="font-size: 3rem;">🌿</div>
            <h3 style="color: #2d9b5a;">Upload a Cotton Leaf Image</h3>
            <p>Supported formats: JPG, JPEG, PNG, WEBP</p>
            <p style="font-size:0.85rem;">For best results: clear image, good lighting, leaf fills the frame</p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# MODE: LIVE WEBCAM
# ════════════════════════════════════════════════════════════════════════════════
elif "Webcam" in mode:
    st.markdown("## 📹 Live Webcam Detection")
    st.info("📌 Position a cotton leaf in front of the camera. Click **Capture & Analyze** to detect disease.")

    col_cam, col_result = st.columns([1, 1], gap="large")

    with col_cam:
        camera_image = st.camera_input("📷 Point camera at cotton leaf")

    with col_result:
        if camera_image:
            image = Image.open(camera_image).convert("RGB")
            st.markdown("### 🔬 Analysis Results")
            run_prediction(predictor, image)
        else:
            st.markdown("""
            <div style="border: 2px dashed #3498db; border-radius: 12px; padding: 2rem;
                        text-align: center; color: #888; height: 300px; display: flex;
                        align-items: center; justify-content: center; flex-direction: column;">
                <div style="font-size: 2.5rem;">📸</div>
                <p>Results will appear here after capture</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    **💡 Tips for accurate webcam detection:**
    - Ensure good lighting (natural daylight preferred)
    - Hold the leaf steady and fill the frame
    - Capture both sides of the leaf if symptoms are on the underside
    - Capture multiple angles for better accuracy
    """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem; color: #aaa; font-size:0.85rem;">
    🌿 <b>CottonGuard AI</b> — Built with PyTorch + Streamlit | 
    For research and advisory use only. Always consult an agronomist for critical decisions.
</div>
""", unsafe_allow_html=True)
