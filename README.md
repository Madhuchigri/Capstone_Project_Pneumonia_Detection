# 🏥 Pneumonia Detection from Chest X-Ray

AI-assisted chest X-ray classification app built with Streamlit, powered by a fine-tuned ResNet50 model trained on the RSNA Pneumonia Detection dataset.

> ⚠️ **Disclaimer:** This tool is for research and educational purposes only. It is **not** a certified medical diagnostic tool and should not be used to make clinical decisions. Always consult a qualified healthcare professional.

---

## Overview

This app lets a user upload a chest X-ray (DICOM or standard image format) and returns a 3-class prediction:

| Class | Meaning |
|-------|---------|
| 🫁 Lung Opacity | Pneumonia likely present |
| ⚠️ No Lung Opacity / Not Normal | Abnormality present, not consistent with typical pneumonia |
| ✅ Normal | No pneumonia detected |

The model is loaded automatically from the Hugging Face Hub at app startup and cached for the session.

---

## Model Details

- **Architecture:** ResNet50 (transfer learning, `imagenet` weights), fine-tuned with the final convolutional block (`conv5`) unfrozen
- **Input size:** 224 × 224 × 3
- **Output:** 3-class softmax
- **Training data:** [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) (2018), ~26,000 chest X-ray DICOM images
- **Model hosting:** [Hugging Face Hub](https://huggingface.co/Maddy2259/Pneumonia_Detection_Model)

---

## Project Structure

```
├── app.py              # Streamlit application entry point
├── requirements.txt    # Python dependencies
├── runtime.txt          # Pinned Python version for deployment
└── README.md
```

---

## Running Locally

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Install dependencies** (Python 3.11 recommended)
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown in the terminal (usually `http://localhost:8501`).

---

## Deployment

This app is deployed on **Streamlit Community Cloud**, pulling the trained model directly from the Hugging Face Hub at runtime rather than storing it in the Git repository.

Key deployment notes:
- `runtime.txt` pins the Python version, since TensorFlow does not yet support the latest Python releases.
- The model file is **not** committed to this repository — it's downloaded from Hugging Face Hub (`huggingface_hub.hf_hub_download`) the first time the app starts, then cached.

---

## Usage

1. Upload a chest X-ray image (`.dcm`, `.png`, `.jpg`, `.jpeg`, `.tiff`, or `.bmp`).
2. The app displays the uploaded image alongside:
   - Primary predicted class with confidence score
   - Certainty metric (based on prediction entropy)
   - Full class probability breakdown
   - A plain-language clinical recommendation note

---

## ⚠️ Important Notes for Maintainers

- **Class order matters.** The label mapping in `app.py` (`CLASS_LABELS`) must exactly match the order produced by `class_label_binarizer.classes_` during training (scikit-learn's `LabelBinarizer` sorts alphabetically by default). Verify this before deploying a newly retrained model.
- **Filename matching.** `MODEL_FILENAME` in `app.py` must exactly match the filename in the Hugging Face repo — filenames are case-sensitive.

---

## Disclaimer

This project was developed as part of an academic capstone project. Predictions from this model should **not** be used for real-world medical diagnosis or treatment decisions.
