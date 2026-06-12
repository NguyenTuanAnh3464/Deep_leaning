"""Bai 1: Train Autoencoder + Classifier cho CIFAR10.

Chay:
    python training/train_cifar10_autoencoder.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
from tensorflow.keras.datasets import cifar10

from utils.labels import CIFAR10_LABELS
from utils.model_builders import build_classifier_from_encoder, build_conv_autoencoder
from utils.preprocess import (
    MODELS_DIR,
    PLOTS_DIR,
    ensure_project_dirs,
    normalize_images,
    plot_loss,
    plot_reconstructions,
    reconstruction_loss,
)


EPOCHS = 5
EPOCHS_LIST = [50, 100, 200]
RUN_EPOCH_COMPARISON = False
BATCH_SIZE = 128
INPUT_SHAPE = (32, 32, 3)


def train_once(epochs=EPOCHS, suffix=""):
    ensure_project_dirs()

    try:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    except Exception as exc:
        print("Khong tai duoc CIFAR10. Hay kiem tra ket noi internet hoac cache TensorFlow/Keras.")
        print(f"Chi tiet loi: {exc}")
        return None

    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    autoencoder, encoder = build_conv_autoencoder(INPUT_SHAPE, loss="mse")
    history_ae = autoencoder.fit(
        x_train,
        x_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
    )

    reconstructed = autoencoder.predict(x_test[:10], verbose=0)
    rec_loss = reconstruction_loss(x_test[:100], autoencoder.predict(x_test[:100], verbose=0))
    print(f"Reconstruction loss CIFAR10: {rec_loss:.6f}")

    plot_loss(history_ae, f"CIFAR10 Autoencoder {epochs} epochs", PLOTS_DIR / f"cifar10_ae_loss{suffix}.png")
    plot_reconstructions(
        x_test[:10],
        reconstructed,
        PLOTS_DIR / f"cifar10_reconstruction{suffix}.png",
        n=10,
    )

    classifier = build_classifier_from_encoder(
        encoder,
        INPUT_SHAPE,
        num_classes=len(CIFAR10_LABELS),
        binary=False,
        freeze_encoder=True,
    )
    history_clf = classifier.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
    )

    test_loss, test_acc = classifier.evaluate(x_test, y_test, verbose=0)
    print(f"Classifier CIFAR10 - test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")

    autoencoder.save(MODELS_DIR / "ae_cifar10_autoencoder.h5")
    encoder.save(MODELS_DIR / "ae_cifar10_encoder.h5")
    classifier.save(MODELS_DIR / "ae_cifar10_classifier.h5")
    plot_loss(history_clf, f"CIFAR10 Classifier {epochs} epochs", PLOTS_DIR / f"cifar10_classifier_loss{suffix}.png")

    print("Da luu model CIFAR10 vao thu muc models/.")
    return {
        "epochs": epochs,
        "reconstruction_loss": rec_loss,
        "validation_loss": float(history_ae.history["val_loss"][-1]),
        "test_accuracy": float(test_acc),
    }


def compare_epochs():
    """Tuy chon so sanh 50, 100, 200 epochs theo yeu cau de bai."""
    results = []
    for epochs in EPOCHS_LIST:
        print(f"\n===== Train CIFAR10 voi epochs = {epochs} =====")
        result = train_once(epochs=epochs, suffix=f"_{epochs}_epochs")
        if result is not None:
            results.append(result)

    print("\nBang so sanh CIFAR10:")
    for item in results:
        print(item)
    print("Nhan xet: epochs tang thuong lam anh tai tao ro hon nhung thoi gian train lau hon.")


def main():
    if RUN_EPOCH_COMPARISON:
        compare_epochs()
    else:
        train_once(EPOCHS)


if __name__ == "__main__":
    main()

