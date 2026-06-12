"""Train ANN phân loại ảnh Cat/Dog.

Ưu tiên dùng tensorflow_datasets cats_vs_dogs. Nếu không tải được, script sẽ
đọc dữ liệu từ:
    datasets/catdog/train/cat
    datasets/catdog/train/dog
    datasets/catdog/val/cat
    datasets/catdog/val/dog
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory

from utils.preprocess import DATASETS_DIR, MODELS_DIR, PLOTS_DIR, ensure_project_dirs, plot_history

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 64


def build_model(input_dim=64 * 64 * 3):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def dataset_to_numpy(dataset, max_batches=None):
    """Chuyển tf.data.Dataset thành numpy array đã flatten."""
    images, labels = [], []
    for index, (batch_x, batch_y) in enumerate(dataset):
        if max_batches is not None and index >= max_batches:
            break
        batch_x = tf.image.resize(batch_x, IMAGE_SIZE)
        batch_x = tf.cast(batch_x, tf.float32) / 255.0
        images.append(tf.reshape(batch_x, (batch_x.shape[0], -1)).numpy())
        labels.append(batch_y.numpy())
    return np.concatenate(images), np.concatenate(labels)


def load_from_tensorflow_datasets():
    import tensorflow_datasets as tfds

    train_ds, val_ds = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:]"],
        as_supervised=True,
        shuffle_files=True,
    )
    train_ds = train_ds.shuffle(2000).batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)
    return dataset_to_numpy(train_ds), dataset_to_numpy(val_ds)


def load_from_folders():
    train_dir = DATASETS_DIR / "catdog" / "train"
    val_dir = DATASETS_DIR / "catdog" / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Không thấy thư mục dữ liệu local. Hãy tạo datasets/catdog/train/cat, "
            "datasets/catdog/train/dog, datasets/catdog/val/cat, datasets/catdog/val/dog."
        )

    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )
    val_ds = image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )
    return dataset_to_numpy(train_ds), dataset_to_numpy(val_ds)


def main():
    ensure_project_dirs()

    try:
        (x_train, y_train), (x_val, y_val) = load_from_tensorflow_datasets()
        print("Đã tải cats_vs_dogs từ tensorflow_datasets.")
    except Exception as exc:
        print("Không tải được cats_vs_dogs từ tensorflow_datasets.")
        print(f"Chi tiết lỗi: {exc}")
        print("Thử đọc dữ liệu từ thư mục datasets/catdog/...")
        try:
            (x_train, y_train), (x_val, y_val) = load_from_folders()
        except Exception as folder_exc:
            print("Không đọc được dữ liệu Cat/Dog từ thư mục local.")
            print(f"Chi tiết lỗi: {folder_exc}")
            return

    model = build_model()
    history = model.fit(
        x_train,
        y_train,
        epochs=8,
        batch_size=64,
        validation_data=(x_val, y_val),
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"Độ chính xác trên tập validation: {val_acc:.4f}")
    print(f"Loss trên tập validation: {val_loss:.4f}")

    model.save(MODELS_DIR / "ann_catdog.h5")
    plot_history(history, "Cat Dog ANN", PLOTS_DIR / "catdog_history.png")
    print("Đã lưu model vào models/ann_catdog.h5")


if __name__ == "__main__":
    main()
