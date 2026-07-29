"""
Streamlit app for Pneumonia Detection. Loads model from HF Hub at startup.
"""
import os
import sys
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.model import build_model

MODEL_REPO = "Maddy2259/Pneumonia_detection_Model"
MODEL_FILENAME = "CNN_base_Model.keras"

CLASS_LABELS = {0: "Normal", 1: "Lung Opacity", 2: "No Lung Opacity / Not Normal"}
CLASS_COLORS = {0: "#28a745", 1: "#dc3545", 2: "#fd7e14"}
CLASS_ICONS  = {0: "✅", 1: "🫁", 2: "⚠️"}

st.set_page_config(
    page_title="Pneumonia Detection from Chest X-Ray",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .pred-card {
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        border-left: 6px solid;
        margin: 1rem 0;
        background: #1e1e2e;
        color: #f0f0f0;
    }
    .pred-label {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    .pred-sub {
        font-size: 0.9rem;
        color: #aaa;
        margin-top: 0.2rem;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-box {
        flex: 1;
        background: #1e1e2e;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
        color: #f0f0f0;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-lbl {
        font-size: 0.75rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .divider { border-top: 1px solid #333; margin: 1.2rem 0; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model from Hugging Face Hub…")
def load_model():
    try:
        weights_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, local_dir="/tmp")
        model = build_model()
        model.load_weights(weights_path)
        return model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None


def preprocess(image: Image.Image) -> np.ndarray:
    import cv2
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
    return np.expand_dims(img.astype(np.float32) / 255.0, axis=0)


def preprocess_dicom(dcm_path: str) -> np.ndarray:
    import cv2, pydicom
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)
    if img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
    return np.expand_dims(img.astype(np.float32) / 255.0, axis=0)


def main():
    st.markdown('<h1 class="main-header">🏥 Pneumonia Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-assisted chest X-ray classification using EfficientNetB3 · 74.76% validation accuracy</p>', unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.warning("Model not available yet. Please train and upload the model first using `run_training.py`.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "This tool classifies chest X-rays into 3 categories using a deep learning model "
            "trained on the RSNA Pneumonia Detection dataset (26,000+ images)."
        )
        st.markdown("### Classes")
        st.markdown("""
| Class | Meaning |
|-------|---------|
| ✅ Normal | No pneumonia detected |
| 🫁 Lung Opacity | Pneumonia likely present |
| ⚠️ Not Normal | Abnormality, not pneumonia |
        """)
        st.markdown("### Model Info")
        st.markdown("- **Architecture:** EfficientNetB3\n- **Input size:** 300×300\n- **Val accuracy:** 74.76%\n- **Dataset:** RSNA 2018")
        st.markdown("---")
        st.caption("⚠️ For research and educational purposes only. Not a medical diagnostic tool.")

    uploaded = st.file_uploader(
        "Upload a Chest X-Ray (DICOM or PNG/JPG)",
        type=["dcm", "png", "jpg", "jpeg", "tiff", "bmp"],
        help="Supports DICOM (.dcm) and standard image formats up to 200MB"
    )

    if uploaded:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("#### Uploaded Image")
            try:
                if uploaded.name.lower().endswith(".dcm"):
                    import tempfile, pydicom, cv2
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    dcm = pydicom.dcmread(tmp_path)
                    arr = dcm.pixel_array
                    if arr.max() > 255:
                        arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
                    st.image(arr, use_container_width=True, caption=f"DICOM · {uploaded.name}")
                    img_input = preprocess_dicom(tmp_path)
                else:
                    image = Image.open(uploaded)
                    st.image(image, use_container_width=True, caption=uploaded.name)
                    img_input = preprocess(image)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                st.stop()

        with col2:
            st.markdown("#### Analysis Results")
            with st.spinner("Running inference…"):
                preds = model.predict(img_input, verbose=0)[0]

            pred_idx = int(np.argmax(preds))
            pred_class = CLASS_LABELS[pred_idx]
            confidence = float(preds[pred_idx])
            color = CLASS_COLORS[pred_idx]
            icon = CLASS_ICONS[pred_idx]

            # Prediction card
            st.markdown(f"""
            <div class="pred-card" style="border-color:{color}">
                <p class="pred-label" style="color:{color}">{icon} {pred_class}</p>
                <p class="pred-sub">Primary prediction with {confidence*100:.1f}% confidence</p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence + entropy metrics
            entropy = float(-np.sum(preds * np.log(preds + 1e-8)))
            max_entropy = float(np.log(3))
            certainty = 1.0 - entropy / max_entropy

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-val" style="color:{color}">{confidence*100:.1f}%</div>
                    <div class="metric-lbl">Confidence</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{certainty*100:.1f}%</div>
                    <div class="metric-lbl">Certainty</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">74.76%</div>
                    <div class="metric-lbl">Model Accuracy</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Class probabilities
            st.markdown("**Class Probabilities**")
            for idx, label in CLASS_LABELS.items():
                bar_color = CLASS_COLORS[idx]
                st.markdown(f"""
                <div style="margin-bottom:0.6rem">
                    <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                        <span style="font-size:0.88rem">{CLASS_ICONS[idx]} {label}</span>
                        <span style="font-size:0.88rem;font-weight:600;color:{bar_color}">{preds[idx]*100:.1f}%</span>
                    </div>
                    <div style="background:#333;border-radius:6px;height:10px">
                        <div style="background:{bar_color};width:{preds[idx]*100:.1f}%;height:10px;border-radius:6px;transition:width 0.3s"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Recommendation
            st.markdown("**Clinical Recommendation**")
            if pred_class == "Lung Opacity":
                st.error("🫁 **Lung opacity detected.** This may indicate pneumonia or other pulmonary conditions. Urgent consultation with a radiologist or physician is strongly recommended.")
            elif pred_class == "No Lung Opacity / Not Normal":
                st.warning("⚠️ **Abnormality detected** but does not appear consistent with typical pneumonia. Further evaluation by a medical professional is recommended.")
            else:
                st.success("✅ **No pneumonia detected.** The X-ray appears normal. Continue routine monitoring if clinically indicated.")

            if confidence < 0.6:
                st.info("ℹ️ Low confidence prediction. The model is uncertain. Please seek professional medical review regardless of result.")

    else:
        st.info("👆 Upload a chest X-ray above to get started.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### ✅ Normal")
            st.markdown("Lungs appear clear with no consolidation or opacity indicating a healthy chest X-ray.")
        with c2:
            st.markdown("#### 🫁 Lung Opacity")
            st.markdown("Presence of opacity or consolidation in the lung fields. May indicate pneumonia or infection.")
        with c3:
            st.markdown("#### ⚠️ Not Normal")
            st.markdown("Abnormality present that is not consistent with typical pneumonia. Requires further evaluation.")


if __name__ == "__main__":
    main()
