"""
AI Crop Disease Detection — Streamlit Web Application

A clean, academic demonstration interface for AI-assisted tomato leaf disease
classification using transfer learning with MobileNetV2.

Calls the existing reusable inference pipeline from src.predict.
"""

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

# ─── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="AI Crop Disease Detection",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for Academic / Clean Aesthetic ────────────────────
st.markdown(
    """
    <style>
    /* Main container and typography */
    .main {
        background-color: #faf9f6;
    }
    h1 {
        color: #1b4332;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #40514e;
        font-size: 1.15rem;
        margin-bottom: 1.5rem;
        line-height: 1.4;
    }
    .note-box {
        background-color: #f1f8f4;
        border-left: 4px solid #2d6a4f;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        color: #2d6a4f;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }
    /* Prediction Cards */
    .result-card {
        background: #ffffff;
        border: 1px solid #e0e5e2;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .result-title {
        color: #1b4332;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .result-conf {
        color: #2d6a4f;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .section-label {
        color: #2b2b2b;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .disclaimer-box {
        background-color: #f7f7f7;
        border-top: 1px solid #e0e0e0;
        padding: 1rem;
        margin-top: 2.5rem;
        font-size: 0.85rem;
        color: #666666;
        text-align: center;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # ─── Sidebar Information ─────────────────────────────────────
    with st.sidebar:
        st.subheader("Project Overview")
        st.write(
            "This application uses **MobileNetV2** transfer learning trained on genuine "
            "PlantVillage tomato leaf images to classify foliar conditions."
        )

        st.markdown("---")
        st.markdown("**Target Crop:** Tomato (*Solanum lycopersicum*)")
        st.markdown("**Supported Classes (4):**")
        for label in CLASS_LABELS.values():
            st.markdown(f"- {label}")

        st.markdown("---")
        st.markdown("**Model Details:**")
        st.markdown("- Base: MobileNetV2 (ImageNet weights)")
        st.markdown("- Input Resolution: 224 × 224 px")
        st.markdown("- Test Set Accuracy: **88.96%**")

    # ─── Main Header ─────────────────────────────────────────────
    st.title("AI-Based Crop Disease Detection")
    st.markdown(
        "<p class='subtitle'>AI-assisted classification of tomato leaf diseases using image analysis.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='note-box'>Upload a clear tomato leaf image to get an AI-assisted prediction.</div>",
        unsafe_allow_html=True,
    )

    # ─── File Uploader ───────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Choose a tomato leaf image...",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

    if uploaded_file is None:
        st.info("Please upload a leaf photograph above to begin analysis.")
        return

    # ─── Display Uploaded Image ──────────────────────────────────
    try:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Uploaded Leaf Image:**")
            st.image(image, use_container_width=True)

        with col2:
            st.write("")
            st.write("")
            st.markdown(
                "Image ready for classification. Click below to run the MobileNetV2 model."
            )
            analyze_button = st.button("Analyze Leaf Image", type="primary", use_container_width=True)

    except Exception as e:
        st.error(f"Unable to process the uploaded image file: {e}")
        return

    # ─── Prediction Execution ────────────────────────────────────
    if analyze_button:
        with st.spinner("Analyzing image features..."):
            try:
                # Seek to start of uploaded buffer
                uploaded_file.seek(0)
                result = predict_image(uploaded_file)
            except Exception as e:
                st.error(f"An unexpected error occurred during inference: {e}")
                return

        if not result.get("success", False):
            st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")
            return

        # ─── Render Results ──────────────────────────────────────
        st.markdown("---")
        st.markdown("### Prediction Results")

        pred_label = result["label"]
        confidence = result["confidence"]
        is_low_conf = result.get("is_low_confidence", False)

        # Main result banner
        st.markdown(
            f"""
            <div class='result-card'>
                <div class='result-title'>{pred_label}</div>
                <div class='result-conf'>Confidence: {confidence * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Low-Confidence Warning if applicable (< 60%)
        if is_low_conf:
            st.warning(
                result.get("warning") or LOW_CONFIDENCE_WARNING
            )

        # Top-3 Predictions
        st.markdown("<div class='section-label'>Top Predictions</div>", unsafe_allow_html=True)
        for pred in result.get("top_3", []):
            rank = pred["rank"]
            name = pred["label"]
            conf_pct = pred["confidence"] * 100
            st.write(f"**{rank}. {name}** — {conf_pct:.1f}%")
            st.progress(float(pred["confidence"]))

        # Disease Information & Recommendations
        disease_info = result.get("disease_info", {})
        if disease_info:
            st.markdown("<div class='section-label'>Symptom Description</div>", unsafe_allow_html=True)
            st.write(disease_info.get("description", "No description available."))

            st.markdown("<div class='section-label'>Recommended Next Steps</div>", unsafe_allow_html=True)
            for rec in disease_info.get("recommendations", []):
                st.markdown(f"- {rec}")

    # ─── Disclaimer ──────────────────────────────────────────────
    st.markdown(
        f"<div class='disclaimer-box'>{DISCLAIMER_TEXT}</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
