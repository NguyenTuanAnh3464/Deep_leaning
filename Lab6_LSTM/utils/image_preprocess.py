"""Ham xu ly anh cho LSTM.

Y tuong: xem moi hang anh la mot time step.
- CIFAR10: 32 time steps, moi step 32 * 3 = 96 dac trung.
- Fashion-MNIST: 28 time steps, moi step 28 dac trung.
- Cat/Dog va Nam/Nu: 64 time steps, moi step 64 * 3 = 192 dac trung.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_project_dirs(base_dir=None):
    """Tao cac thu muc can thiet neu chua co."""
    base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
    folders = [
        "datasets",
        "datasets/catdog/train/cat",
        "datasets/catdog/train/dog",
        "datasets/catdog/val/cat",
        "datasets/catdog/val/dog",
        "datasets/gender/train/male",
        "datasets/gender/train/female",
        "datasets/gender/val/male",
        "datasets/gender/val/female",
        "models",
        "tokenizers",
        "static/uploads",
    ]
    for folder in folders:
        (base_dir / folder).mkdir(parents=True, exist_ok=True)


def cifar10_to_sequence(images):
    """Chuyen anh CIFAR10 tu (n, 32, 32, 3) sang (n, 32, 96)."""
    images = images.astype("float32") / 255.0
    return images.reshape((images.shape[0], 32, 32 * 3))


def fashion_mnist_to_sequence(images):
    """Chuyen anh Fashion-MNIST tu (n, 28, 28) sang (n, 28, 28)."""
    return images.astype("float32") / 255.0


def rgb64_to_sequence(images):
    """Chuyen anh mau 64x64x3 sang sequence (n, 64, 192)."""
    images = images.astype("float32") / 255.0
    return images.reshape((images.shape[0], 64, 64 * 3))


def load_image_as_sequence(image_path, model_type):
    """Doc mot anh moi va bien doi dung shape dau vao cua tung model."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Khong tim thay anh: {image_path}")

    if model_type == "fashion_mnist":
        img = Image.open(image_path).convert("L").resize((28, 28))
        arr = np.array(img).astype("float32") / 255.0
        return arr.reshape((1, 28, 28)), img

    if model_type == "cifar10":
        img = Image.open(image_path).convert("RGB").resize((32, 32))
        arr = np.array(img).astype("float32") / 255.0
        return arr.reshape((1, 32, 96)), img

    if model_type in {"catdog", "gender"}:
        img = Image.open(image_path).convert("RGB").resize((64, 64))
        arr = np.array(img).astype("float32") / 255.0
        return arr.reshape((1, 64, 192)), img

    raise ValueError("model_type phai la cifar10, fashion_mnist, catdog hoac gender")


def has_image_data(folder):
    """Kiem tra thu muc co anh hay khong."""
    folder = Path(folder)
    if not folder.exists():
        return False
    return any(path.suffix.lower() in IMAGE_EXTENSIONS for path in folder.rglob("*"))


def build_image_generators(train_dir, val_dir, target_size=(64, 64), batch_size=32):
    """Tao generator doc anh tu thu muc train/val."""
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )
    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
    return train_gen, val_gen


def sequence_generator(image_generator):
    """Bien doi batch anh (n, 64, 64, 3) thanh (n, 64, 192) cho LSTM."""
    while True:
        x_batch, y_batch = next(image_generator)
        x_batch = x_batch.reshape((x_batch.shape[0], 64, 64 * 3))
        yield x_batch, y_batch


def plot_history(history, title, save_path=None):
    """Ve bieu do loss va accuracy neu history co accuracy."""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("loss", []), label="train loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val loss")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("accuracy", []), label="train acc")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val acc")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def show_sample_images(images, labels, class_names, n=9):
    """Hien thi mot vai anh mau."""
    n = min(n, len(images))
    plt.figure(figsize=(8, 8))
    for i in range(n):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i], cmap="gray" if images[i].ndim == 2 else None)
        label_id = int(np.squeeze(labels[i]))
        plt.title(class_names[label_id])
        plt.axis("off")
    plt.tight_layout()
    plt.show()
