# ASL Alphabet Recognition (A–Z) — Real-Time Webcam CNN (TensorFlow/Keras)

A production-style computer vision project that trains a **Convolutional Neural Network (CNN)** to recognize **American Sign Language (ASL) fingerspelling letters A–Z** from hand images and runs **real-time inference** on a webcam feed with on-screen predictions + confidence.

**Dataset:** ASL Alphabet Dataset (Kaggle, `khansatehreem`)  
https://www.kaggle.com/datasets/khansatehreem/asl-alphabet-dataset

> Scope: This repository classifies **single letters (A–Z)** from **static frames**. It does **not** translate full words/sentences.

---

## Highlights

- Real-time webcam demo (OpenCV overlay: predicted letter + confidence)
- Custom CNN with BatchNorm + Dropout regularization
- Data augmentation + early stopping + learning-rate scheduling
- Reproducible exports: best model, label encoder, training curves, confusion matrix

---

## Results (held-out test)

- **Test Accuracy:** ~**99.9%**
- **Macro F1:** ~**0.999**
- **Test Loss:** ~**0.01**

Artifacts:
- `models/training_curves.png`
- `models/confusion_matrix.png`

> Evaluation note: Metrics are reported on a **strictly held-out** test set that is **never used** during training, tuning, or augmentation.

---

## Dataset

- Source: Kaggle ASL Alphabet dataset (`A`–`Z` folders).
- Classes: **26** (A–Z). *(If your dataset version includes extra labels like `space/del/nothing`, this repo filters to A–Z.)*
- Preprocessing:
  - Resize to **64×64**
  - Convert to **grayscale** (`64×64×1`)
  - Normalize pixel values to `[0, 1]`

Recommended split:
- Train/Val: from `data/train` (e.g., 85/15)
- Test: `data/test` kept **strictly held-out**

---

## Model

Custom CNN:

- Input: `64×64×1`
- 3 conv blocks: **32 → 64 → 128** filters, `3×3`, `same`, ReLU
- Regularization: **Batch Normalization + Dropout (0.25–0.50)**
- Classifier: `Dense(256, ReLU)` + Dropout + `Dense(26, softmax)`

Training:
- Optimizer: **Adam** (lr=1e-3) + `ReduceLROnPlateau`
- Loss: categorical cross-entropy
- Early stopping: monitor validation loss
- Augmentation: rotation, shift, zoom, shear

(Implementation: `src/model_utils.py`, `src/data_utils.py`)

---

## Project structure

```text
sign_language_project/
  data/
    train/        # A–Z folders (training source)
    test/         # A–Z folders (held-out evaluation)
  models/
    best_sign_model.h5
    label_encoder.joblib
    training_curves.png
    confusion_matrix.png
  src/
    data_utils.py
    model_utils.py
    train.py
    inference_webcam.py
  requirements.txt
  README.md

```

----

## How to run

# 1. set up

git clone <your-repo-url>
cd sign_language_project

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download Data

Download the dataset from Kaggle and unzip into:

data/
  train/
  test/

Do not move/copy test images into training data. Keep data/test as a true hold-out set.

# 3. Train + export artifacts

python src/train.py

This exports:

- models/best_sign_model.h5
- models/label_encoder.joblib
- models/training_curves.png
- models/confusion_matrix.png

# 4. Real-time webcam inference

python src/inference_webcam.py

- A webcam window opens
- Show an ASL letter (A–Z) in frame
- The predicted letter + confidence is displayed on-screen
Press q to quit. or ctrl c

---
## Reproducibility

To make results repeatable:

- Fix random seeds (NumPy + TensorFlow)
- Log dataset counts + split sizes
- Export artifacts (model + label encoder + plots)

If you maintain an experiment log, store it under reports/ and reference it here.

---

## Limitations

High accuracy on this dataset may not fully reflect real-world performance because:

- backgrounds/lighting can be cleaner than uncontrolled environments
- camera angle/hand scale/skin tone coverage may be limited
- ASL letters J and Z are dynamic in real signing; single-frame classification is a simplification

---

## Future work

- Transfer learning baseline (MobileNetV2 / EfficientNet) with latency–accuracy comparison
- Add “no-hand/unknown” rejection via confidence thresholding + calibration (ECE)
- Dynamic signs (J, Z) using video or keypoints (MediaPipe Hands)
- Deploy as a Streamlit or FastAPI app

---
## Acknowledgements

- Kaggle dataset: khansatehreem/asl-alphabet-dataset


If you want this to look even more “industry,” replace the `~99.9%` placeholders with **your actual printed metrics** and add one line under Results stating **exact test set size** (e.g., “26×800 images = 20,800”). That single detail boosts credibility a lot.
