import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from utils.image_preprocess import (
    build_image_generators,
    ensure_project_dirs,
    has_image_data,
    plot_history,
    sequence_generator,
)

EPOCHS = 3
BATCH_SIZE = 32
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_catdog.h5"
LOCAL_TRAIN_DIR = PROJECT_ROOT / "datasets" / "catdog" / "train"
LOCAL_VAL_DIR = PROJECT_ROOT / "datasets" / "catdog" / "val"


def build_model():
    model = Sequential(
        [
            LSTM(128, input_shape=(64, 192)),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def preprocess_tfds(image, label):
    image = tf.image.resize(image, (64, 64))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, (64, 192))
    return image, tf.cast(label, tf.float32)


def load_tfds_data():
    import tensorflow_datasets as tfds

    print("Thu tai dataset cats_vs_dogs bang tensorflow_datasets...")
    train_ds, val_ds = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:]"],
        as_supervised=True,
    )
    train_ds = train_ds.map(preprocess_tfds).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_tfds).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, None, None


def load_local_data():
    if not has_image_data(LOCAL_TRAIN_DIR) or not has_image_data(LOCAL_VAL_DIR):
        print("Khong co dataset Cat/Dog cuc bo.")
        print("Hay dat anh vao:")
        print("datasets/catdog/train/cat, datasets/catdog/train/dog")
        print("datasets/catdog/val/cat, datasets/catdog/val/dog")
        return None

    train_gen, val_gen = build_image_generators(
        LOCAL_TRAIN_DIR,
        LOCAL_VAL_DIR,
        target_size=(64, 64),
        batch_size=BATCH_SIZE,
    )
    train_seq = sequence_generator(train_gen)
    val_seq = sequence_generator(val_gen)
    return train_seq, val_seq, train_gen, val_gen


def main():
    ensure_project_dirs()
    model = build_model()
    model.summary()

    try:
        train_data, val_data, train_gen, val_gen = load_tfds_data()
    except Exception as exc:
        print(f"Khong tai duoc cats_vs_dogs bang tensorflow_datasets: {exc}")
        local_data = load_local_data()
        if local_data is None:
            return
        train_data, val_data, train_gen, val_gen = local_data

    fit_kwargs = {"validation_data": val_data, "epochs": EPOCHS}
    if train_gen is not None and val_gen is not None:
        fit_kwargs["steps_per_epoch"] = max(1, train_gen.samples // BATCH_SIZE)
        fit_kwargs["validation_steps"] = max(1, val_gen.samples // BATCH_SIZE)

    history = model.fit(train_data, **fit_kwargs)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Da luu model vao: {MODEL_PATH}")
    plot_history(history, "LSTM Cat/Dog", PROJECT_ROOT / "static" / "catdog_history.png")


if __name__ == "__main__":
    main()

