"""Bai 2: Train Autoencoder + Classifier cho Cat/Dog.

Uu tien dung tensorflow_datasets dataset "cats_vs_dogs".
Neu khong tai duoc, dat anh theo cau truc:
    datasets/catdog/train/cat
    datasets/catdog/train/dog
    datasets/catdog/val/cat
    datasets/catdog/val/dog

Chay:
    python training/train_catdog_autoencoder.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import tensorflow as tf

from utils.labels import CATDOG_LABELS
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

# De train nhanh khi test code. Doi thanh None neu muon train toan bo dataset.
MAX_TRAIN_BATCHES = 200
MAX_VAL_BATCHES = 50


def normalize_batch(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.float32)
    return image, label


def to_autoencoder_pair(image, label):
    return image, image


def limit_dataset(dataset, max_batches):
    return dataset.take(max_batches) if max_batches is not None else dataset


def folder_has_images(folder):
    image_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    for pattern in image_exts:
        if any(folder.glob(pattern)):
            return True
    return False


def load_from_local_folder():
    base_dir = DATASETS_DIR / "catdog"
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    required_dirs = [train_dir / "cat", train_dir / "dog", val_dir / "cat", val_dir / "dog"]

    if not all(folder.exists() and folder_has_images(folder) for folder in required_dirs):
        print("Chua co dataset Cat/Dog trong thu muc datasets/catdog.")
        print("Hay dat anh vao:")
        print("  datasets/catdog/train/cat, datasets/catdog/train/dog")
        print("  datasets/catdog/val/cat, datasets/catdog/val/dog")
        return None, None

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=CATDOG_LABELS,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        class_names=CATDOG_LABELS,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.cast(y, tf.float32)))
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.cast(y, tf.float32)))
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)


def load_catdog_dataset():
    try:
        import tensorflow_datasets as tfds

        train_raw, val_raw = tfds.load(
            "cats_vs_dogs",
            split=["train[:80%]", "train[80%:]"],
            as_supervised=True,
        )
        train_ds = (
            train_raw.shuffle(2000)
            .map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            val_raw.map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
        return limit_dataset(train_ds, MAX_TRAIN_BATCHES), limit_dataset(val_ds, MAX_VAL_BATCHES)
    except Exception as exc:
        print("Khong tai duoc tensorflow_datasets cats_vs_dogs, chuyen sang doc thu muc local.")
        print(f"Chi tiet loi: {exc}")
        return load_from_local_folder()


def train_once(epochs=EPOCHS, suffix=""):
    ensure_project_dirs()
    train_ds, val_ds = load_catdog_dataset()
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
    print(f"Reconstruction loss Cat/Dog: {rec_loss:.6f}")

    plot_loss(history_ae, f"Cat/Dog Autoencoder {epochs} epochs", PLOTS_DIR / f"catdog_ae_loss{suffix}.png")
    plot_reconstructions(
        sample_images[:10].numpy(),
        reconstructed,
        PLOTS_DIR / f"catdog_reconstruction{suffix}.png",
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
    print(f"Classifier Cat/Dog - validation loss: {val_loss:.4f}, validation accuracy: {val_acc:.4f}")

    autoencoder.save(MODELS_DIR / "ae_catdog_autoencoder.h5")
    encoder.save(MODELS_DIR / "ae_catdog_encoder.h5")
    classifier.save(MODELS_DIR / "ae_catdog_classifier.h5")
    plot_loss(history_clf, f"Cat/Dog Classifier {epochs} epochs", PLOTS_DIR / f"catdog_classifier_loss{suffix}.png")

    print("Da luu model Cat/Dog vao thu muc models/.")
    return {
        "epochs": epochs,
        "reconstruction_loss": rec_loss,
        "validation_loss": float(history_ae.history["val_loss"][-1]),
        "validation_accuracy": float(val_acc),
    }


def compare_epochs():
    results = []
    for epochs in EPOCHS_LIST:
        print(f"\n===== Train Cat/Dog voi epochs = {epochs} =====")
        result = train_once(epochs=epochs, suffix=f"_{epochs}_epochs")
        if result is not None:
            results.append(result)

    print("\nBang so sanh Cat/Dog:")
    for item in results:
        print(item)
    print("Nhan xet: Cat/Dog phu thuoc nhieu vao chat luong va do da dang cua dataset.")


def main():
    if RUN_EPOCH_COMPARISON:
        compare_epochs()
    else:
        train_once(EPOCHS)


if __name__ == "__main__":
    main()
