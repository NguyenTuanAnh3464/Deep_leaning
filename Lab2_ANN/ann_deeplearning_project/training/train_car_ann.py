"""Train ANN dự báo chất lượng xe Car Evaluation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

from utils.labels import CAR_FEATURES
from utils.preprocess import MODELS_DIR, PLOTS_DIR, ensure_project_dirs, plot_history

CAR_COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]


def load_car_dataset():
    """Tải Car Evaluation từ ucimlrepo, nếu lỗi thì đọc trực tiếp URL UCI."""
    try:
        from ucimlrepo import fetch_ucirepo

        car = fetch_ucirepo(id=19)
        x = car.data.features.copy()
        y = car.data.targets.copy()
        df = pd.concat([x, y], axis=1)
        target_col = y.columns[0]
        df = df.rename(columns={target_col: "class"})
        return df
    except Exception as exc:
        print("Không tải được Car Evaluation bằng ucimlrepo, thử đọc URL UCI.")
        print(f"Chi tiết lỗi: {exc}")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    return pd.read_csv(url, names=CAR_COLUMNS)


def build_model(input_dim, num_classes=4):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def dense_array(data):
    """Đổi sparse matrix thành numpy array vì Keras dễ train hơn với dense input."""
    return data.toarray() if hasattr(data, "toarray") else data


def main():
    ensure_project_dirs()

    try:
        df = load_car_dataset()
    except Exception as exc:
        print("Không tải được dataset Car Evaluation. Hãy kiểm tra kết nối internet.")
        print(f"Chi tiết lỗi: {exc}")
        return

    x = df[CAR_FEATURES]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["class"])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), CAR_FEATURES),
        ]
    )
    x_train_processed = dense_array(preprocessor.fit_transform(x_train))
    x_test_processed = dense_array(preprocessor.transform(x_test))

    model = build_model(x_train_processed.shape[1], len(label_encoder.classes_))
    history = model.fit(
        x_train_processed,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test_processed, y_test, verbose=0)
    print(f"Độ chính xác trên tập test: {test_acc:.4f}")
    print(f"Loss trên tập test: {test_loss:.4f}")

    model.save(MODELS_DIR / "ann_car.h5")
    joblib.dump(preprocessor, MODELS_DIR / "car_preprocessor.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "car_label_encoder.joblib")
    plot_history(history, "Car Evaluation ANN", PLOTS_DIR / "car_history.png")
    print("Đã lưu model vào models/ann_car.h5")
    print("Đã lưu encoder vào models/car_preprocessor.joblib và models/car_label_encoder.joblib")


if __name__ == "__main__":
    main()
