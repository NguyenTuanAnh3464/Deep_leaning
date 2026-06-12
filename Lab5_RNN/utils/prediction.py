from __future__ import annotations

import numpy as np
from joblib import load

from utils.data_loader import get_dataset_config


class MissingModelError(FileNotFoundError):
    """Loi khi chua co model hoac scaler da train."""


def parse_recent_values(raw_values: str) -> list[float]:
    """Chuyen chuoi '1, 2, 3' thanh danh sach so thuc."""
    values = []
    for item in raw_values.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def predict_next_value(dataset_key: str, recent_values: list[float], time_steps: int = 12) -> float:
    """Du bao gia tri tiep theo dua tren N gia tri gan nhat."""
    config = get_dataset_config(dataset_key)
    model_path = config["model_path"]
    scaler_path = config["scaler_path"]

    if not model_path.exists() or not scaler_path.exists():
        raise MissingModelError("Chưa có model. Vui lòng chạy file train tương ứng trước.")

    if len(recent_values) != time_steps:
        raise ValueError(f"Cần nhập đúng {time_steps} giá trị gần nhất, ngăn cách bằng dấu phẩy.")

    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise RuntimeError("Chưa cài tensorflow. Hãy chạy: pip install -r requirements.txt") from exc

    model = load_model(str(model_path), compile=False)
    scaler = load(scaler_path)

    arr = np.array(recent_values, dtype=float).reshape(-1, 1)
    scaled = scaler.transform(arr)
    X = scaled.reshape(1, time_steps, 1)
    prediction_scaled = model.predict(X, verbose=0)
    prediction = scaler.inverse_transform(prediction_scaled)
    return float(prediction[0][0])
