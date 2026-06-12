import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

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
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_gender.h5"
TRAIN_DIR = PROJECT_ROOT / "datasets" / "gender" / "train"
VAL_DIR = PROJECT_ROOT / "datasets" / "gender" / "val"


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


def main():
    ensure_project_dirs()
    if not has_image_data(TRAIN_DIR) or not has_image_data(VAL_DIR):
        print("Chua co dataset khuon mat Nam/Nu.")
        print(
            "Vui long tai dataset khuon mat Nam/Nu public nhu FairFace, "
            "sau do tach anh vao thu muc male/female."
        )
        print("Vi du:")
        print("datasets/gender/train/male, datasets/gender/train/female")
        print("datasets/gender/val/male, datasets/gender/val/female")
        return

    train_gen, val_gen = build_image_generators(
        TRAIN_DIR,
        VAL_DIR,
        target_size=(64, 64),
        batch_size=BATCH_SIZE,
    )
    train_seq = sequence_generator(train_gen)
    val_seq = sequence_generator(val_gen)

    model = build_model()
    model.summary()
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        steps_per_epoch=max(1, train_gen.samples // BATCH_SIZE),
        validation_steps=max(1, val_gen.samples // BATCH_SIZE),
        epochs=EPOCHS,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Da luu model vao: {MODEL_PATH}")
    plot_history(history, "LSTM Nam/Nu", PROJECT_ROOT / "static" / "gender_history.png")


if __name__ == "__main__":
    main()

