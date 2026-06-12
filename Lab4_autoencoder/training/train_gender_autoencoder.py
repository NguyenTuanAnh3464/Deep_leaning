"""Bai 4: Train Autoencoder + Classifier cho anh khuon mat Nam/Nu.

Dat dataset theo cau truc:
    datasets/gender/train/male
    datasets/gender/train/female
    datasets/gender/val/male
    datasets/gender/val/female

Co the dung dataset public nhu FairFace, sau do tach anh vao cac thu muc tren.

Chay:
    python training/train_gender_autoencoder.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import tensorflow as tf

from utils.labels import GENDER_LABELS
from utils.model_builders import build_classifier_from_encoder, build_conv_autoencoder
from utils.preprocess import (
    DATASETS_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    ensure_project_dirs,
    plot_loss,
    plot_reconstructions,
    reconstruction_loss,
)


EPOCHS = 5
EPOCHS_LIST = [50, 100, 200]
RUN_EPOCH_COMPARISON = False
BATCH_SIZE = 32
IMAGE_SIZE = (64, 64)
INPUT_SHAPE = (64, 64, 3)
FOLDER_CLASS_NAMES = ["female", "male"]


def folder_has_images(folder):
    image_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    for pattern in image_exts:
        if any(folder.glob(pattern)):
            return True
    return False


def to_autoencoder_pair(image, label):
    return image, image


def load_gender_dataset():
    base_dir = DATASETS_DIR / "gender"
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    required_dirs = [train_dir / "female", train_dir / "male", val_dir / "female", val_dir / "male"]

    if not all(folder.exists() and folder_has_images(folder) for folder in required_dirs):
        print("Vui long tai dataset khuon mat Nam/Nu public nhu FairFace,")
        print("sau do tach anh vao thu muc male/female theo cau truc:")
        print("  datasets/gender/train/male, datasets/gender/train/female")
        print("  datasets/gender/val/male, datasets/gender/val/female")
        return None, None

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=FOLDER_CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        class_names=FOLDER_CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.cast(y, tf.float32)))
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.cast(y, tf.float32)))
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)


def train_once(epochs=EPOCHS, suffix=""):
    ensure_project_dirs()
    train_ds, val_ds = load_gender_dataset()
    if train_ds is None or val_ds is None:
        return None

    train_ae = train_ds.map(to_autoencoder_pair)
    val_ae = val_ds.map(to_autoencoder_pair)

    autoencoder, encoder = build_conv_autoencoder(INPUT_SHAPE, loss="mse")
    history_ae = autoencoder.fit(
        train_ae,
        epochs=epochs,
        validation_data=val_ae,
        verbose=1,
    )

    sample_images, _ = next(iter(val_ds))
    reconstructed = autoencoder.predict(sample_images[:10], verbose=0)
    rec_loss = reconstruction_loss(sample_images[:10].numpy(), reconstructed)
    print(f"Reconstruction loss Nam/Nu: {rec_loss:.6f}")

    plot_loss(history_ae, f"Nam/Nu Autoencoder {epochs} epochs", PLOTS_DIR / f"gender_ae_loss{suffix}.png")
    plot_reconstructions(
        sample_images[:10].numpy(),
        reconstructed,
        PLOTS_DIR / f"gender_reconstruction{suffix}.png",
        n=10,
    )

    classifier = build_classifier_from_encoder(
        encoder,
        INPUT_SHAPE,
        num_classes=2,
        binary=True,
        freeze_encoder=True,
    )
    history_clf = classifier.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=1,
    )

    val_loss, val_acc = classifier.evaluate(val_ds, verbose=0)
    print(f"Classifier Nam/Nu - validation loss: {val_loss:.4f}, validation accuracy: {val_acc:.4f}")

    autoencoder.save(MODELS_DIR / "ae_gender_autoencoder.h5")
    encoder.save(MODELS_DIR / "ae_gender_encoder.h5")
    classifier.save(MODELS_DIR / "ae_gender_classifier.h5")
    plot_loss(history_clf, f"Nam/Nu Classifier {epochs} epochs", PLOTS_DIR / f"gender_classifier_loss{suffix}.png")

    print("Da luu model Nam/Nu vao thu muc models/.")
    return {
        "epochs": epochs,
        "reconstruction_loss": rec_loss,
        "validation_loss": float(history_ae.history["val_loss"][-1]),
        "validation_accuracy": float(val_acc),
    }


def compare_epochs():
    results = []
    for epochs in EPOCHS_LIST:
        print(f"\n===== Train Nam/Nu voi epochs = {epochs} =====")
        result = train_once(epochs=epochs, suffix=f"_{epochs}_epochs")
        if result is not None:
            results.append(result)

    print("\nBang so sanh Nam/Nu:")
    for item in results:
        print(item)
    print("Nhan xet: Bai Nam/Nu phu thuoc rat nhieu vao chat luong, do can bang va cach tach dataset.")


def main():
    if RUN_EPOCH_COMPARISON:
        compare_epochs()
    else:
        train_once(EPOCHS)


if __name__ == "__main__":
    main()
