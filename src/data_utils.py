import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 64

def load_data_dir(data_dir, allowed_classes=None):
    data = []
    labels = []
    classes = sorted(os.listdir(data_dir))
    for folder in classes:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        if allowed_classes is not None and folder not in allowed_classes:
            continue

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(folder)

    data = np.array(data, dtype="float32") / 255.0
    data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)
    return data, labels

def prepare_datasets(train_dir, test_dir, val_size=0.15, random_state=42):
    # Load train first to determine which classes exist (A–M in this dataset)
    X_train_full, y_train_full = load_data_dir(train_dir)
    train_classes = sorted(list(set(y_train_full)))

    # Load test but only for those classes
    X_test, y_test = load_data_dir(test_dir, allowed_classes=train_classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        stratify=y_train_full,
        random_state=random_state
    )

    encoder = LabelBinarizer()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)
    y_test_enc = encoder.transform(y_test)

    return (X_train, y_train_enc,
            X_val, y_val_enc,
            X_test, y_test_enc,
            encoder)

def create_generators(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1
    )

    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_gen = val_datagen.flow(X_val, y_val, batch_size=batch_size)
    test_gen = test_datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=False)

    return train_gen, val_gen, test_gen
