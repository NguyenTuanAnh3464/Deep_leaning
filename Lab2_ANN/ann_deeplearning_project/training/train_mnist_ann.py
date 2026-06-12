"""Train ANN nhận dạng chữ số viết tay MNIST."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

from utils.preprocess import MODELS_DIR, PLOTS_DIR, ensure_project_dirs, normalize_and_flatten_images, plot_history


def build_model(input_dim=784, num_classes=10):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(256, activation="relu"),
            Dropout(0.2),
            Dense(128, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    ensure_project_dirs()

    try:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    except Exception as exc:
        print("Không tải được MNIST. Hãy kiểm tra kết nối internet hoặc cache TensorFlow.")
        print(f"Chi tiết lỗi: {exc}")
        return

    x_train = normalize_and_flatten_images(x_train)
    x_test = normalize_and_flatten_images(x_test)

    model = build_model()
    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Độ chính xác trên tập test: {test_acc:.4f}")
    print(f"Loss trên tập test: {test_loss:.4f}")

    model.save(MODELS_DIR / "ann_mnist.h5")
    plot_history(history, "MNIST ANN", PLOTS_DIR / "mnist_history.png")
    print("Đã lưu model vào models/ann_mnist.h5")


if __name__ == "__main__":
    main()
