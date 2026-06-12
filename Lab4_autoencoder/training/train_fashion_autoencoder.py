"""Bai 3: Train Autoencoder + Classifier cho Fashion-MNIST.

Chay:
    python training/train_fashion_autoencoder.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from tensorflow.keras.datasets import fashion_mnist

from utils.labels import FASHION_MNIST_LABELS
from utils.model_builders import build_classifier_from_encoder, build_dense_autoencoder
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
INPUT_SHAPE = (28, 28, 1)
LATENT_DIM = 32


def train_once(epochs=EPOCHS, suffix=""):
    ensure_project_dirs()

    try:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    except Exception as exc:
        print("Khong tai duoc Fashion-MNIST. Hay kiem tra ket noi internet hoac cache TensorFlow/Keras.")
        print(f"Chi tiet loi: {exc}")
        return None

    x_train = normalize_images(x_train, grayscale=True)
    x_test = normalize_images(x_test, grayscale=True)

    autoencoder, encoder = build_dense_autoencoder(INPUT_SHAPE, latent_dim=LATENT_DIM, loss="mse")
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
    print(f"Reconstruction loss Fashion-MNIST: {rec_loss:.6f}")

    plot_loss(history_ae, f"Fashion-MNIST Autoencoder {epochs} epochs", PLOTS_DIR / f"fashion_ae_loss{suffix}.png")
    plot_reconstructions(
        x_test[:10],
        reconstructed,
        PLOTS_DIR / f"fashion_reconstruction{suffix}.png",
        n=10,
        grayscale=True,
    )

    classifier = build_classifier_from_encoder(
        encoder,
        INPUT_SHAPE,
        num_classes=len(FASHION_MNIST_LABELS),
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
    print(f"Classifier Fashion-MNIST - test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")

    autoencoder.save(MODELS_DIR / "ae_fashion_autoencoder.h5")
    encoder.save(MODELS_DIR / "ae_fashion_encoder.h5")
    classifier.save(MODELS_DIR / "ae_fashion_classifier.h5")
    plot_loss(history_clf, f"Fashion-MNIST Classifier {epochs} epochs", PLOTS_DIR / f"fashion_classifier_loss{suffix}.png")

    print("Da luu model Fashion-MNIST vao thu muc models/.")
    return {
        "epochs": epochs,
        "reconstruction_loss": rec_loss,
        "validation_loss": float(history_ae.history["val_loss"][-1]),
        "test_accuracy": float(test_acc),
    }


def compare_epochs():
    results = []
    for epochs in EPOCHS_LIST:
        print(f"\n===== Train Fashion-MNIST voi epochs = {epochs} =====")
        result = train_once(epochs=epochs, suffix=f"_{epochs}_epochs")
        if result is not None:
            results.append(result)

    print("\nBang so sanh Fashion-MNIST:")
    for item in results:
        print(item)
    print("Nhan xet: Fashion-MNIST don gian hon CIFAR10 nen Autoencoder thuong tai tao anh ro hon.")


def main():
    if RUN_EPOCH_COMPARISON:
        compare_epochs()
    else:
        train_once(EPOCHS)


if __name__ == "__main__":
    main()

