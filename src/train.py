import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model

from data_utils import prepare_datasets, create_generators, IMG_SIZE
from model_utils import build_cnn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def plot_history(history, out_dir):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close()

def main():
    (X_train, y_train,
     X_val, y_val,
     X_test, y_test,
     encoder) = prepare_datasets(TRAIN_DIR, TEST_DIR)

    num_classes = y_train.shape[1]
    print("Classes:", encoder.classes_)
    print("Num classes:", num_classes)

    train_gen, val_gen, test_gen = create_generators(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64
    )

    model = build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1),
                      num_classes=num_classes, lr=1e-3)

    checkpoint_path = os.path.join(MODELS_DIR, "best_sign_model.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    lr_reducer = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_gen,
        epochs=40,
        validation_data=val_gen,
        callbacks=[checkpoint, lr_reducer, early_stop]
    )

    plot_history(history, MODELS_DIR)

    best_model = load_model(checkpoint_path)

    test_loss, test_acc = best_model.evaluate(test_gen, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    y_prob = best_model.predict(test_gen)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    class_names = encoder.classes_

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"))
    plt.close()

    encoder_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    joblib.dump(encoder, encoder_path)
    print(f"Saved encoder to {encoder_path}")

if __name__ == "__main__":
    main()
