"""Tiền xử lý dữ liệu và hàm dùng chung.

Các hàm trong file này cố ý viết đơn giản để sinh viên dễ đọc, dễ chỉnh khi
thực hành trên máy cá nhân hoặc Google Colab.
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
    """Tạo các thư mục cần thiết nếu chưa tồn tại."""
    for folder in [MODELS_DIR, DATASETS_DIR, UPLOAD_DIR, PLOTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def normalize_and_flatten_images(images):
    """Chuẩn hóa pixel về [0, 1] và flatten ảnh thành vector 1 chiều."""
    images = images.astype("float32") / 255.0
    return images.reshape((images.shape[0], -1))


def preprocess_image_for_model(image_path, image_size, color_mode="rgb"):
    """Đọc ảnh upload, resize, chuẩn hóa và flatten để đưa vào ANN."""
    mode = "L" if color_mode == "grayscale" else "RGB"
    image = Image.open(image_path).convert(mode).resize(image_size)
    array = np.asarray(image).astype("float32") / 255.0
    return array.reshape(1, -1)


def plot_history(history, title, output_path):
    """Vẽ biểu đồ accuracy và loss sau khi train."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train accuracy")
    if val_acc:
        plt.plot(val_acc, label="Validation accuracy")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train loss")
    if val_loss:
        plt.plot(val_loss, label="Validation loss")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_upload(file_storage, upload_folder=UPLOAD_DIR):
    """Lưu file upload từ Flask vào static/uploads."""
    upload_folder = Path(upload_folder)
    upload_folder.mkdir(parents=True, exist_ok=True)
    filename = Path(file_storage.filename).name
    save_path = upload_folder / filename
    file_storage.save(save_path)
    return save_path
