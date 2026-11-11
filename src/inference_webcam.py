import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model

from data_utils import IMG_SIZE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_sign_model.keras")
LEGACY_MODEL_PATH = os.path.join(MODELS_DIR, "best_sign_model.h5")  # models trained before the .keras switch
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
ROI_FRACTION = 0.6  # side of the centred hand box, as a fraction of the shorter frame side

def main():
    model = load_model(MODEL_PATH if os.path.exists(MODEL_PATH) else LEGACY_MODEL_PATH)
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

        # Classify only a centred square: the training images are close-ups of a hand,
        # so feeding the whole frame (face, background) hurts accuracy.
        frame = cv2.flip(frame, 1)  # mirror view, easier to sign into
        h, w = frame.shape[:2]
        side = int(min(h, w) * ROI_FRACTION)
        x0, y0 = (w - side) // 2, (h - side) // 2
        roi = frame[y0:y0 + side, x0:x0 + side]
        cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (255, 200, 0), 2)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        gray = gray.astype("float32") / 255.0
        gray = np.expand_dims(gray, axis=(0, -1))

        preds = model(gray, training=False).numpy()  # faster than predict() for one frame
        idx = np.argmax(preds[0])
        label = class_names[idx]
        conf = preds[0][idx]

        # Draw label at top-left
        cv2.putText(frame, f"{label} ({conf:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow(f"ASL Alphabet Recognition ({class_names[0]}–{class_names[-1]})", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
