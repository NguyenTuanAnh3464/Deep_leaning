import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from utils.image_preprocess import cifar10_to_sequence, ensure_project_dirs, plot_history, show_sample_images
from utils.labels import CIFAR10_LABELS

EPOCHS = 3
BATCH_SIZE = 128
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_cifar10.h5"


def build_model():
    model = Sequential(
        [
            LSTM(128, input_shape=(32, 96), return_sequences=False),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax"),
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
    print("Dang tai dataset CIFAR10 tu tensorflow.keras.datasets...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    show_sample_images(x_train, y_train, CIFAR10_LABELS, n=9)
    x_train_seq = cifar10_to_sequence(x_train)
    x_test_seq = cifar10_to_sequence(x_test)

    model = build_model()
    model.summary()

    history = model.fit(
        x_train_seq,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    loss, acc = model.evaluate(x_test_seq, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Da luu model vao: {MODEL_PATH}")

    pred = model.predict(x_test_seq[:1], verbose=0)
    print("Nhan that:", CIFAR10_LABELS[int(y_test[0][0])])
    print("Nhan du doan:", CIFAR10_LABELS[int(tf.argmax(pred[0]))])

    plot_history(history, "LSTM CIFAR10", PROJECT_ROOT / "static" / "cifar10_history.png")


if __name__ == "__main__":
    main()

