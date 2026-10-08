"""
AI Crop Disease Detection — Streamlit Web Application

A botanical frosted-glass interface for AI-assisted tomato leaf disease
classification using transfer learning with MobileNetV2.

Calls the existing reusable inference pipeline from src.predict.
"""

import base64
import io
import sys
from pathlib import Path
from PIL import Image
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASS_LABELS,
    CONFIDENCE_THRESHOLD,
    DISCLAIMER_TEXT,
    DISEASE_INFO,
    LOW_CONFIDENCE_WARNING,
)
from src.predict import predict_image

# ─── Load Reference Botanical Background ─────────────────────────
BG_IMAGE_PATH = PROJECT_ROOT / "assets" / "bg_botanical.jpg"
if BG_IMAGE_PATH.exists():
    with open(BG_IMAGE_PATH, "rb") as f:
        BG_BASE64 = base64.b64encode(f.read()).decode()
else:
    BG_BASE64 = ""

# ─── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="AI Crop Disease Detection",
   
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark Botanical + Frosted Glass CSS ──────────────────────────
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* ══════════════════════════════════════════════════════════════
       GLOBAL BACKGROUND: EXACT BOTANICAL FOLIAGE
       ══════════════════════════════════════════════════════════════ */
    html, body, .stApp, [data-testid="stAppViewContainer"] {{
        background: url('data:image/jpeg;base64,{BG_BASE64}') no-repeat center center fixed !important;
        background-size: cover !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        color: #e4eee4 !important;
        padding-top: 0 !important;
        margin-top: 0 !important;
    }}

    /* Global overlay for balanced contrast */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: radial-gradient(circle at 50% 50%, rgba(5, 12, 6, 0.4) 0%, rgba(3, 8, 4, 0.7) 100%);
        pointer-events: none;
        z-index: 0;
    }}

    /* Zero out all default Streamlit header and padding */
    header[data-testid="stHeader"] {{
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
    }}

    [data-testid="stAppViewContainer"],
    .main,
    section.main,
    [data-testid="stAppViewBlockContainer"] {{
        padding-top: 0 !important;
        margin-top: 0 !important;
    }}

    #MainMenu, footer, [data-testid="stDeployButton"] {{
        display: none !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       SIDEBAR: FROSTED TRANSLUCENT GLASS
       ══════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {{
        background: rgba(10, 20, 12, 0.75) !important;
        backdrop-filter: blur(24px) saturate(1.3) !important;
        -webkit-backdrop-filter: blur(24px) saturate(1.3) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 10px 0 35px rgba(0, 0, 0, 0.4) !important;
        z-index: 10 !important;
    }}

    [data-testid="stSidebar"] * {{
        color: #d6e4d6 !important;
    }}

    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
    }}

    [data-testid="stSidebar"] strong {{
        color: #bfe0bf !important;
    }}

    [data-testid="stSidebar"] hr {{
        border-color: rgba(255, 255, 255, 0.1) !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       THE SINGLE BIG FROSTED GLASS CONTAINER (CENTERED DASHBOARD)
       ══════════════════════════════════════════════════════════════ */
    .main .block-container {{
        max-width: 950px !important;
        margin: 4.5vh auto !important;
        padding: 2.2rem 2.8rem 1.8rem 2.8rem !important;
        background: rgba(14, 28, 16, 0.65) !important;
        backdrop-filter: blur(24px) saturate(1.35) !important;
        -webkit-backdrop-filter: blur(24px) saturate(1.35) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 28px !important;
        box-shadow:
            0 25px 65px rgba(0, 0, 0, 0.55),
            inset 0 1px 1px rgba(255, 255, 255, 0.25),
            0 0 100px rgba(20, 50, 24, 0.2) !important;
        position: relative !important;
        z-index: 2 !important;
    }}

    /* Top glass edge reflection */
    .main .block-container::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 8%;
        right: 8%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.45), transparent);
        border-radius: 28px 28px 0 0;
    }}

    /* ══════════════════════════════════════════════════════════════
       HERO TYPOGRAPHY (CENTERED DASHBOARD TITLE)
       ══════════════════════════════════════════════════════════════ */
    .hero-kicker {{
        font-size: 0.76rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #a4cca4;
        font-weight: 500;
        margin-bottom: 0.3rem;
        text-align: center;
    }}

    .hero-title {{
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.15;
        color: #ffffff;
        letter-spacing: -0.01em;
        text-align: center;
        margin: 0 auto 0.4rem auto;
        text-shadow: 0 4px 18px rgba(0, 0, 0, 0.4);
    }}


    .section-title {{
        font-size: 1.15rem;
        font-weight: 600;
        color: #f2f7f2;
        margin-top: 0;
        margin-bottom: 0.25rem;
        letter-spacing: 0.01em;
    }}

    .thin-divider {{
        width: 100%;
        height: 1px;
        background: rgba(255, 255, 255, 0.16);
        margin: 0.5rem 0 0.9rem 0;
    }}

    .body-desc {{
        font-size: 0.88rem;
        line-height: 1.5;
        color: #d1dfd1;
        font-weight: 300;
        margin-bottom: 0.7rem;
    }}

    /* ══════════════════════════════════════════════════════════════
       PILL BUTTONS, BADGES & TOOLTIPS
       ══════════════════════════════════════════════════════════════ */
    .pill-badge {{
        display: inline-block;
        padding: 0.32rem 0.95rem;
        border-radius: 30px;
        font-size: 0.76rem;
        letter-spacing: 0.03em;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        background: rgba(25, 45, 28, 0.55);
        color: #e4eee4;
        border: 1px solid rgba(255, 255, 255, 0.16);
        backdrop-filter: blur(8px);
        transition: all 0.22s ease;
    }}

    .pill-badge:hover {{
        background: rgba(40, 70, 45, 0.75);
        border-color: rgba(255, 255, 255, 0.35);
        color: #ffffff;
    }}

    .pill-badge-highlight {{
        background: #eef5ec !important;
        color: #122415 !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.25) !important;
    }}

    /* Tooltip Pop-up */
    .pill-tooltip {{
        position: relative;
        display: inline-block;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }}

    .pill-tooltip .pill-badge {{
        margin: 0 !important;
        cursor: pointer;
    }}

    .pill-tooltip .tooltip-bubble {{
        visibility: hidden;
        opacity: 0;
        width: 230px;
        background: rgba(10, 22, 12, 0.96);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.22);
        color: #e2ede2;
        text-align: left;
        border-radius: 12px;
        padding: 0.65rem 0.85rem;
        font-size: 0.75rem;
        line-height: 1.45;
        position: absolute;
        z-index: 1000;
        bottom: 130%;
        left: 50%;
        transform: translateX(-50%) translateY(4px);
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        transition: opacity 0.2s ease, visibility 0.2s ease, transform 0.2s ease;
        pointer-events: none;
    }}

    .pill-tooltip .tooltip-bubble strong {{
        display: block;
        color: #a4cca4;
        font-size: 0.78rem;
        margin-bottom: 0.2rem;
        letter-spacing: 0.02em;
    }}

    .pill-tooltip:hover .tooltip-bubble {{
        visibility: visible;
        opacity: 1;
        transform: translateX(-50%) translateY(0);
    }}

    .pill-tooltip .tooltip-bubble::after {{
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: rgba(10, 22, 12, 0.96) transparent transparent transparent;
    }}



    /* ══════════════════════════════════════════════════════════════
       IMAGE PREVIEW & UPLOADER
       ══════════════════════════════════════════════════════════════ */
    .preview-glass-box {{
        background: rgba(10, 22, 12, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.3);
        text-align: center;
        backdrop-filter: blur(10px);
    }}

    [data-testid="stFileUploader"] {{
        background: rgba(16, 32, 18, 0.45) !important;
        border: 1px dashed rgba(255, 255, 255, 0.25) !important;
        border-radius: 16px !important;
        padding: 0.5rem 0.8rem !important;
        margin-bottom: 0.6rem !important;
        backdrop-filter: blur(8px) !important;
    }}

    [data-testid="stFileUploader"] section {{
        background: transparent !important;
        padding: 0.2rem 0 !important;
    }}

    [data-testid="stFileUploader"] label {{
        color: #d0ded0 !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
    }}

    [data-testid="stFileUploader"] small {{
        color: #9cb89c !important;
        font-size: 0.74rem !important;
    }}

    [data-testid="stFileUploader"] button {{
        background: #eef5ec !important;
        color: #122415 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.35rem 1.1rem !important;
        font-size: 0.8rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    }}

    /* Action Trigger Button */
    .stButton > button {{
        background: #eef5ec !important;
        color: #122415 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.03em !important;
        border-radius: 25px !important;
        padding: 0.55rem 1.8rem !important;
        border: none !important;
        box-shadow: 0 5px 16px rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        width: 100% !important;
    }}

    .stButton > button:hover {{
        background: #ffffff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.4) !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       PREDICTION CARD: INNER GLASS HIGHLIGHT
       ══════════════════════════════════════════════════════════════ */
    .prediction-highlight {{
        background: rgba(22, 42, 25, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        margin: 0.8rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }}

    .pred-badge {{
        font-size: 0.72rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #9cd19c;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }}

    .pred-name {{
        font-size: 1.65rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.15rem;
    }}

    .pred-confidence {{
        font-size: 1.02rem;
        color: #b4dfb4;
        font-weight: 500;
    }}

    /* Progress bars */
    .stProgress > div > div {{
        background: rgba(15, 30, 18, 0.45) !important;
        border-radius: 10px !important;
        height: 6px !important;
    }}

    .stProgress > div > div > div {{
        background: linear-gradient(90deg, #74b374, #a3d9a3) !important;
        border-radius: 10px !important;
    }}

    /* Image Display */
    .main img {{
        border-radius: 14px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
    }}

    /* Disclaimer & Warnings */
    .disclaimer-strip {{
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 0.6rem;
        margin-top: 0.8rem;
        font-size: 0.72rem;
        line-height: 1.45;
        color: rgba(190, 215, 190, 0.6);
        text-align: center;
    }}

    [data-testid="stAlert"] {{
        background: rgba(35, 45, 20, 0.6) !important;
        border: 1px solid rgba(200, 180, 80, 0.35) !important;
        border-radius: 12px !important;
        color: #f0ebd0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # ─── Sidebar: Technical Metadata ──────────────────────────────
    with st.sidebar:
        st.markdown("<div class='hero-kicker'>INTELLIGENCE SPECS</div>", unsafe_allow_html=True)
        st.markdown("### System Architecture")
        st.write(
            "Transfer learning model with **MobileNetV2** backbone trained on genuine "
            "PlantVillage tomato foliage images."
        )

        st.markdown("---")
        st.markdown("**Target Crop:** Tomato (*Solanum lycopersicum*)")
        st.markdown("**Classes Monitored (4):**")
        for label in CLASS_LABELS.values():
            st.markdown(f"- {label}")

        st.markdown("---")
        st.markdown("**Performance Metrics:**")
        st.markdown("- Input Resolution: 224 x 224 px")
        st.markdown("- Test Set Accuracy: **88.96%**")
        st.markdown("- Macro Precision: **89.08%**")
        st.markdown("- Macro Recall: **88.50%**")

    # ══════════════════════════════════════════════════════════════
    # INSIDE THE SINGLE BIG FROSTED GLASS CONTAINER
    # ══════════════════════════════════════════════════════════════

    # Header Row (Centered dashboard header)
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 0.3rem;">
            <div class="hero-kicker">ABOUT CROP LIFE & PATHOLOGY</div>
            <h1 class="hero-title">Leaf Disease AI</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='thin-divider' style='margin: 0.5rem auto 1.2rem auto;'></div>", unsafe_allow_html=True)


    # Two-Column Content Layout
    col_left, col_right = st.columns([1.05, 1.35], gap="large")

    with col_left:
        st.markdown("<div class='section-title'>Foliage Sample</div>", unsafe_allow_html=True)
        st.markdown(
            "<p class='body-desc'>Upload a leaf photograph to analyze foliar condition.</p>",
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload tomato leaf image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="Supported: JPG, JPEG, PNG",
        )

        if uploaded_file is not None:
            try:
                uploaded_image = Image.open(uploaded_file)
                st.image(uploaded_image, use_container_width=True)
                st.write("")
                analyze_button = st.button("Analyze Foliage", type="primary")
            except Exception as e:
                st.error(f"Cannot process image file: {e}")
                analyze_button = False
        else:
            # Elegant compact default placeholder when waiting for upload
            st.markdown(
                """
                <div class="preview-glass-box" style="padding: 1.3rem 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.4rem;">🌿</div>
                    <div style="font-weight: 500; color: #ffffff; font-size: 0.95rem; margin-bottom: 0.2rem;">No Leaf Uploaded</div>
                    <div style="font-size: 0.8rem; color: #a4cca4;">Select a tomato leaf photo above to begin classification</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            analyze_button = False

    with col_right:
        if uploaded_file is not None and analyze_button:
            with st.spinner("Classifying leaf pathology with MobileNetV2..."):
                try:
                    uploaded_file.seek(0)
                    result = predict_image(uploaded_file)
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    result = None

            if result and result.get("success", False):
                pred_label = result["label"]
                confidence = result["confidence"]
                is_low_conf = result.get("is_low_confidence", False)

                # Diagnostic result banner
                st.markdown(
                    f"""
                    <div class="prediction-highlight">
                        <div class="pred-badge">PREDICTED CONDITION</div>
                        <div class="pred-name">{pred_label}</div>
                        <div class="pred-confidence">AI Confidence: {confidence * 100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if is_low_conf:
                    st.warning(result.get("warning") or LOW_CONFIDENCE_WARNING)

                # Top-3 Probabilities
                st.markdown("<div class='section-title' style='font-size: 1.05rem;'>Probability Distribution</div>", unsafe_allow_html=True)
                for pred in result.get("top_3", []):
                    rank = pred["rank"]
                    name = pred["label"]
                    conf_pct = pred["confidence"] * 100
                    st.write(f"**{rank}. {name}** — {conf_pct:.1f}%")
                    st.progress(float(pred["confidence"]))

                # Disease Details & Agricultural Advice
                disease_info = result.get("disease_info", {})
                if disease_info:
                    st.markdown("<div class='thin-divider' style='margin: 0.6rem 0;'></div>", unsafe_allow_html=True)
                    st.markdown("<div class='section-title' style='font-size: 1.05rem;'>Symptom Details</div>", unsafe_allow_html=True)
                    st.markdown(f"<p class='body-desc' style='margin-bottom: 0.6rem;'>{disease_info.get('description', 'N/A')}</p>", unsafe_allow_html=True)

                    st.markdown("<div class='section-title' style='font-size: 1.05rem;'>Recommended Crop Care</div>", unsafe_allow_html=True)
                    for rec in disease_info.get("recommendations", []):
                        st.markdown(f"<span style='color: #d1dfd1; font-size: 0.85rem;'>&bull; {rec}</span><br>", unsafe_allow_html=True)
            elif result:
                st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")

        else:
            # Default guide display matching reference image's editorial feel
            st.markdown("<div class='section-title'>Diagnostic Overview</div>", unsafe_allow_html=True)
            st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <p class='body-desc' style='margin-bottom: 0.6rem;'>
                    Early detection of foliar symptoms protects harvest yields.
                    This vision pipeline identifies pathogen lesions caused by
                    fungal and oomycete vectors in tomato plants.
                </p>
                """,
                unsafe_allow_html=True,
            )

            # Class badge pills with hover tooltips
            st.markdown(
                """
                <div style='margin-bottom: 0.8rem;'>
                    <div class='pill-tooltip'>
                        <span class='pill-badge'>Healthy Foliage</span>
                        <div class='tooltip-bubble'>
                            <strong>Healthy Foliage</strong>
                            Normal, disease-free tomato leaf with uniform green coloration, clear veins, and no necrotic lesions.
                        </div>
                    </div>
                    <div class='pill-tooltip'>
                        <span class='pill-badge'>Early Blight</span>
                        <div class='tooltip-bubble'>
                            <strong>Early Blight (Alternaria solani)</strong>
                            Fungal infection causing dark brown spots with concentric ring patterns ("target-board") and yellow halos.
                        </div>
                    </div>
                    <div class='pill-tooltip'>
                        <span class='pill-badge'>Late Blight</span>
                        <div class='tooltip-bubble'>
                            <strong>Late Blight (Phytophthora infestans)</strong>
                            Aggressive water-mold pathogen causing rapidly expanding dark, water-soaked foliar lesions in cool, humid weather.
                        </div>
                    </div>
                    <div class='pill-tooltip'>
                        <span class='pill-badge'>Leaf Mold</span>
                        <div class='tooltip-bubble'>
                            <strong>Leaf Mold (Passalora fulva)</strong>
                            Foliar fungus causing pale yellow chlorotic spots on upper leaf surfaces and velvety olive-green mold underneath.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="prediction-highlight" style="background: rgba(18, 32, 20, 0.4); padding: 0.9rem 1.1rem; margin: 0.4rem 0;">
                    <div class="pred-badge">HOW IT WORKS</div>
                    <div style="font-size: 0.85rem; line-height: 1.55; color: #c4d6c4;">
                        1. Upload a clear photograph of a single tomato leaf.<br>
                        2. Click <strong>Analyze Foliage</strong> to trigger the neural network.<br>
                        3. Review disease classification, confidence, and recommended agricultural treatments.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Footer Action Pills
    st.markdown("<div class='thin-divider' style='margin-top: 1rem; margin-bottom: 0.7rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span class="pill-badge">🌿 Solanum lycopersicum</span>
                <span class="pill-badge">🔬 MobileNetV2</span>
                <span class="pill-badge">🎯 88.96% Accuracy</span>
                <span class="pill-badge">📊 PlantVillage Dataset</span>
            </div>
            <div style="font-size: 0.78rem; color: #a4cca4; font-weight: 500;">
                Crafted for healthy crops. 🍃
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scientific Disclaimer
    st.markdown(
        f"<div class='disclaimer-strip'>{DISCLAIMER_TEXT}</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
