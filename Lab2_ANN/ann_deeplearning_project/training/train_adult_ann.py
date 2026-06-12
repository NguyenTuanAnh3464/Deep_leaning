"""Train ANN dự báo thu nhập Adult Income."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

from utils.labels import ADULT_CATEGORICAL_FEATURES, ADULT_FEATURES, ADULT_NUMERIC_FEATURES
from utils.preprocess import MODELS_DIR, PLOTS_DIR, ensure_project_dirs, plot_history


ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def load_adult_dataset():
    """Tải Adult từ ucimlrepo, nếu lỗi thì đọc trực tiếp từ URL UCI."""
    try:
        from ucimlrepo import fetch_ucirepo

        adult = fetch_ucirepo(id=2)
        x = adult.data.features.copy()
        y = adult.data.targets.copy()
        df = pd.concat([x, y], axis=1)
        target_col = y.columns[0]
        df = df.rename(columns={target_col: "income"})
        return df
    except Exception as exc:
        print("Không tải được Adult bằng ucimlrepo, thử đọc URL UCI.")
        print(f"Chi tiết lỗi: {exc}")

    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    ]
    frames = []
    for url in urls:
        frame = pd.read_csv(
            url,
            names=ADULT_COLUMNS,
            na_values=[" ?", "?"],
            skipinitialspace=True,
            comment="|",
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_model(input_dim):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def dense_array(data):
    """Đổi sparse matrix thành numpy array vì Keras dễ train hơn với dense input."""
    return data.toarray() if hasattr(data, "toarray") else data


def main():
    ensure_project_dirs()

    try:
        df = load_adult_dataset()
    except Exception as exc:
        print("Không tải được dataset Adult. Hãy kiểm tra kết nối internet.")
        print(f"Chi tiết lỗi: {exc}")
        return

    df = df.replace("?", np.nan)
    df["income"] = df["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    df = df[df["income"].isin(["<=50K", ">50K"])]

    x = df[ADULT_FEATURES]
    y = (df["income"] == ">50K").astype("int32")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, ADULT_NUMERIC_FEATURES),
            ("cat", categorical_pipeline, ADULT_CATEGORICAL_FEATURES),
        ]
    )

    x_train_processed = dense_array(preprocessor.fit_transform(x_train))
    x_test_processed = dense_array(preprocessor.transform(x_test))

    model = build_model(x_train_processed.shape[1])
    history = model.fit(
        x_train_processed,
        y_train,
        epochs=15,
        batch_size=128,
        validation_split=0.1,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test_processed, y_test, verbose=0)
    print(f"Độ chính xác trên tập test: {test_acc:.4f}")
    print(f"Loss trên tập test: {test_loss:.4f}")

    model.save(MODELS_DIR / "ann_adult.h5")
    joblib.dump(preprocessor, MODELS_DIR / "adult_preprocessor.joblib")
    plot_history(history, "Adult Income ANN", PLOTS_DIR / "adult_history.png")
    print("Đã lưu model vào models/ann_adult.h5")
    print("Đã lưu bộ tiền xử lý vào models/adult_preprocessor.joblib")


if __name__ == "__main__":
    main()
