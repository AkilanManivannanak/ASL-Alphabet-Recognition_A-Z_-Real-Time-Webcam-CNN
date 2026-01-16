import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model

from data_utils import IMG_SIZE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_sign_model.h5")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")

def main():
    model = load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    class_names = encoder.classes_

    # Try 0; if needed, change to 1 or 2
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame from camera.")
            break

        # Use full frame as ROI (no cropping)
        roi = frame

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        gray = gray.astype("float32") / 255.0
        gray = np.expand_dims(gray, axis=(0, -1))

        preds = model.predict(gray, verbose=0)
        idx = np.argmax(preds[0])
        label = class_names[idx]
        conf = preds[0][idx]

        # Draw label at top-left
        cv2.putText(frame, f"{label} ({conf:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("ASL Alphabet Recognition (A–M)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
