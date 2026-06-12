from __future__ import annotations

import numpy as np
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils.data_loader import PLOT_DIR, get_dataset_config, get_target_series


def create_sequences(data: np.ndarray, time_steps: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Tao du lieu dang sequence cho RNN: X=(samples, time_steps, 1), y=(samples,)."""
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps : i, 0])
        y.append(data[i, 0])
    X = np.array(X).reshape(-1, time_steps, 1)
    y = np.array(y)
    return X, y


def build_simple_rnn_model(time_steps: int = 12):
    """Xay dung mo hinh SimpleRNN dung theo yeu cau bai thuc hanh."""
    try:
        from tensorflow.keras.layers import Dense, SimpleRNN
        from tensorflow.keras.models import Sequential
    except ImportError as exc:
        raise RuntimeError("Chua cai tensorflow. Hay chay: pip install -r requirements.txt") from exc

    model = Sequential(
        [
            SimpleRNN(64, activation="tanh", input_shape=(time_steps, 1)),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_rnn_for_dataset(
    dataset_key: str,
    time_steps: int = 12,
    epochs: int = 20,
    batch_size: int = 16,
    save_plots: bool = True,
    verbose: int = 1,
) -> dict:
    """Train RNN cho mot dataset, luu model/scaler va tra ve metric."""
    config = get_dataset_config(dataset_key)
    series = get_target_series(dataset_key)
    values = series.values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)

    X, y = create_sequences(scaled_values, time_steps=time_steps)
    if len(X) < 10:
        raise ValueError("So sequence qua it. Hay giam time_steps hoac bo sung du lieu.")

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = build_simple_rnn_model(time_steps=time_steps)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    pred_scaled = model.predict(X_test, verbose=0)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    pred_real = scaler.inverse_transform(pred_scaled).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_test_real, pred_real)))
    mae = float(mean_absolute_error(y_test_real, pred_real))

    config["model_path"].parent.mkdir(parents=True, exist_ok=True)
    config["scaler_path"].parent.mkdir(parents=True, exist_ok=True)
    model.save(str(config["model_path"]))
    dump(scaler, config["scaler_path"])

    plot_paths = {}
    if save_plots:
        plot_paths = save_training_plots(config, history, y_test_real, pred_real)

    return {
        "dataset_key": dataset_key,
        "title": config["title"],
        "dataset": config["dataset_name"],
        "target": config["target"],
        "time_steps": time_steps,
        "model": "SimpleRNN",
        "rmse": rmse,
        "mae": mae,
        "model_path": str(config["model_path"]),
        "scaler_path": str(config["scaler_path"]),
        "history": history,
        "actual": y_test_real,
        "prediction": pred_real,
        "plot_paths": plot_paths,
    }


def split_train_test_sequences(series, time_steps: int = 12, test_size: float = 0.2):
    """Ham rieng de notebook co the minh hoa qua trinh chia train/test."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(np.asarray(series).reshape(-1, 1))
    X, y = create_sequences(scaled, time_steps=time_steps)
    return train_test_split(X, y, test_size=test_size, shuffle=False), scaler


def save_training_plots(config: dict, history, actual: np.ndarray, prediction: np.ndarray) -> dict:
    """Luu bieu do loss va Actual vs Prediction vao static/training_plots/."""
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    loss_path = PLOT_DIR / f"{config['plot_prefix']}_loss.png"
    pred_path = PLOT_DIR / f"{config['plot_prefix']}_prediction.png"

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title(f"Loss - {config['display_name']}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=120)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(actual, label="Actual")
    plt.plot(prediction, label="Prediction")
    plt.title(f"Actual vs Prediction - {config['display_name']}")
    plt.xlabel("Mau test")
    plt.ylabel(config["target"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(pred_path, dpi=120)
    plt.close()

    return {"loss": str(loss_path), "prediction": str(pred_path)}
