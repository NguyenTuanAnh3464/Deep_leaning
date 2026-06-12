"""Tien xu ly anh, ve bieu do va du doan anh moi.

Tat ca duong dan deu la duong dan tuong doi tinh tu project, phu hop khi chay
tren may ca nhan hoac Google Colab.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
UPLOAD_DIR = PROJECT_ROOT / "static" / "uploads"
PLOTS_DIR = PROJECT_ROOT / "static" / "training_plots"


def ensure_project_dirs():
    """Tao cac thu muc can thiet neu chua co."""
    for folder in [MODELS_DIR, DATASETS_DIR, UPLOAD_DIR, PLOTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def normalize_images(images, grayscale=False):
    """Chuan hoa pixel ve khoang [0, 1] va them kenh mau neu can."""
    images = images.astype("float32") / 255.0
    if grayscale and images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    return images


def preprocess_image_for_classifier(image_path, target_size, grayscale=False):
    """Doc anh, resize, chuan hoa va them batch dimension."""
    mode = "L" if grayscale else "RGB"
    image = Image.open(image_path).convert(mode).resize(target_size)
    array = np.asarray(image).astype("float32") / 255.0
    if grayscale:
        array = np.expand_dims(array, axis=-1)
    return np.expand_dims(array, axis=0)


def reconstruction_loss(x_true, x_pred):
    """Tinh reconstruction loss MSE trung binh."""
    return float(np.mean(np.square(x_true - x_pred)))


def plot_loss(history, title, output_path):
    """Ve train loss va validation loss cua Autoencoder hoac classifier."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(7, 4))
    plt.plot(loss, label="Train loss")
    if val_loss:
        plt.plot(val_loss, label="Validation loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_reconstructions(original, reconstructed, output_path, n=10, grayscale=False):
    """Luu hinh so sanh anh goc va anh tai tao."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(n, len(original), len(reconstructed))
    plt.figure(figsize=(n * 1.5, 3))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        if grayscale:
            plt.imshow(original[i].squeeze(), cmap="gray")
        else:
            plt.imshow(np.clip(original[i], 0, 1))
        plt.axis("off")
        if i == 0:
            plt.ylabel("Goc")

        plt.subplot(2, n, n + i + 1)
        if grayscale:
            plt.imshow(reconstructed[i].squeeze(), cmap="gray")
        else:
            plt.imshow(np.clip(reconstructed[i], 0, 1))
        plt.axis("off")
        if i == 0:
            plt.ylabel("Tai tao")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def predict_image_autoencoder(classifier_path, image_path, target_size, class_names, grayscale=False):
    """Load classifier, doc anh moi va in nhan du doan kem do tin cay."""
    from tensorflow.keras.models import load_model

    classifier_path = Path(classifier_path)
    if not classifier_path.exists():
        raise FileNotFoundError("Chua co model. Vui long chay file train tuong ung truoc.")

    model = load_model(classifier_path, compile=False)
    x = preprocess_image_for_classifier(image_path, target_size, grayscale=grayscale)
    prediction = model.predict(x, verbose=0)

    if len(class_names) == 2 and prediction.shape[-1] == 1:
        positive_probability = float(prediction[0][0])
        index = 1 if positive_probability >= 0.5 else 0
        confidence = positive_probability if index == 1 else 1.0 - positive_probability
    else:
        index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

    label = class_names[index]
    print(f"Nhãn dự đoán: {label}")
    print(f"Độ tin cậy: {confidence * 100:.2f}%")

    image = Image.open(image_path)
    plt.imshow(image, cmap="gray" if grayscale else None)
    plt.title(f"{label} - {confidence * 100:.2f}%")
    plt.axis("off")
    plt.show()

    return label, confidence

