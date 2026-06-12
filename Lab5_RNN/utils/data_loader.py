from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets"
MODEL_DIR = PROJECT_ROOT / "models"
SCALER_DIR = PROJECT_ROOT / "scalers"
PLOT_DIR = PROJECT_ROOT / "static" / "training_plots"


DATASET_CONFIG = {
    "house": {
        "title": "Dự báo giá nhà",
        "display_name": "Dự báo giá nhà",
        "dataset_name": "raw_sales.csv",
        "path": DATASET_DIR / "raw_sales.csv",
        "target": "price",
        "model_path": MODEL_DIR / "rnn_house_price.h5",
        "scaler_path": SCALER_DIR / "scaler_house_price.pkl",
        "plot_prefix": "house_price",
    },
    "btc": {
        "title": "Dự báo giá Bitcoin",
        "display_name": "Dự báo giá Bitcoin",
        "dataset_name": "BTC_DATA.csv",
        "path": DATASET_DIR / "BTC_DATA.csv",
        "target": "priceUSD",
        "model_path": MODEL_DIR / "rnn_btc_price.h5",
        "scaler_path": SCALER_DIR / "scaler_btc_price.pkl",
        "plot_prefix": "btc_price",
    },
    "voltage": {
        "title": "Dự báo điện thế Voltage",
        "display_name": "Dự báo Voltage",
        "dataset_name": "household_power_consumption.txt",
        "path": DATASET_DIR / "household_power_consumption.txt",
        "target": "Voltage",
        "model_path": MODEL_DIR / "rnn_voltage.h5",
        "scaler_path": SCALER_DIR / "scaler_voltage.pkl",
        "plot_prefix": "voltage",
    },
    "nifty": {
        "title": "Dự báo giá đóng cửa NIFTY",
        "display_name": "Dự báo NIFTY close",
        "dataset_name": "NIFTY_stock_market.csv",
        "path": DATASET_DIR / "NIFTY_stock_market.csv",
        "target": "close",
        "model_path": MODEL_DIR / "rnn_nifty_close.h5",
        "scaler_path": SCALER_DIR / "scaler_nifty_close.pkl",
        "plot_prefix": "nifty_close",
    },
}


def ensure_project_dirs() -> None:
    """Tao cac thu muc can thiet neu chua co."""
    for folder in [DATASET_DIR, MODEL_DIR, SCALER_DIR, PLOT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def get_dataset_config(dataset_key: str) -> dict:
    """Lay cau hinh cua mot bai du bao."""
    if dataset_key not in DATASET_CONFIG:
        keys = ", ".join(DATASET_CONFIG)
        raise ValueError(f"dataset_key khong hop le. Hay chon mot trong: {keys}")
    return DATASET_CONFIG[dataset_key]


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def download_yfinance(symbol: str, output_path: Path, target_col: str, period: str = "5y") -> pd.DataFrame:
    """Tai du lieu tai chinh bang yfinance va doi cot Close thanh cot muc tieu."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Chua cai yfinance. Hay chay: pip install -r requirements.txt") from exc

    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    df = _flatten_yfinance_columns(df)
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"Khong tai duoc du lieu {symbol} tu yfinance.")

    output = df.reset_index()[["Date", "Close"]].rename(columns={"Close": target_col})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def download_house_public(output_path: Path) -> pd.DataFrame:
    """Tai dataset gia nha public tu GitHub va chuan hoa cot median_house_value thanh price."""
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    if "median_house_value" not in df.columns:
        raise RuntimeError("Dataset gia nha public khong co cot median_house_value.")

    output = pd.DataFrame(
        {
            "date": pd.date_range("2010-01-01", periods=len(df), freq="D"),
            "price": df["median_house_value"].astype(float),
        }
    )
    output.to_csv(output_path, index=False)
    return output


def download_household_power(output_path: Path, max_rows: int = 5000) -> pd.DataFrame:
    """Tai dataset Individual Household Electric Power Consumption tu UCI."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open("household_power_consumption.txt") as file_obj:
            df = pd.read_csv(file_obj, sep=";", na_values=["?"], nrows=max_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep=";", index=False)
    return df


def create_demo_dataset(dataset_key: str, rows: int = 240) -> pd.DataFrame:
    """Tao du lieu mau nho de sinh vien co the chay thu khi chua tai duoc dataset public."""
    rng = np.random.default_rng(42)
    t = np.arange(rows)

    if dataset_key == "house":
        values = 180000 + 900 * t + 25000 * np.sin(t / 10) + rng.normal(0, 9000, rows)
        df = pd.DataFrame({"date": pd.date_range("2018-01-01", periods=rows, freq="D"), "price": values})
    elif dataset_key == "btc":
        values = 8000 + 110 * t + 1800 * np.sin(t / 8) + rng.normal(0, 700, rows)
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=rows, freq="D"), "priceUSD": values})
    elif dataset_key == "voltage":
        values = 235 + 4 * np.sin(t / 6) + rng.normal(0, 1.2, rows)
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2021-01-01", periods=rows, freq="D").strftime("%d/%m/%Y"),
                "Time": "00:00:00",
                "Global_active_power": rng.uniform(0.2, 5.0, rows),
                "Global_reactive_power": rng.uniform(0.0, 0.5, rows),
                "Voltage": values,
                "Global_intensity": rng.uniform(1.0, 20.0, rows),
                "Sub_metering_1": rng.uniform(0.0, 5.0, rows),
                "Sub_metering_2": rng.uniform(0.0, 5.0, rows),
                "Sub_metering_3": rng.uniform(0.0, 20.0, rows),
            }
        )
    elif dataset_key == "nifty":
        values = 10000 + 35 * t + 350 * np.sin(t / 11) + rng.normal(0, 150, rows)
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=rows, freq="D"), "close": values})
    else:
        raise ValueError(f"Khong ho tro dataset_key: {dataset_key}")

    config = get_dataset_config(dataset_key)
    if dataset_key == "voltage":
        df.to_csv(config["path"], sep=";", index=False)
    else:
        df.to_csv(config["path"], index=False)
    return df


def ensure_dataset(dataset_key: str, allow_demo: bool = True) -> pd.DataFrame:
    """Dam bao dataset ton tai: uu tien file co san, sau do tai public, cuoi cung tao demo."""
    ensure_project_dirs()
    config = get_dataset_config(dataset_key)
    path = config["path"]

    if path.exists() and path.stat().st_size > 0:
        return load_dataset(dataset_key)

    try:
        if dataset_key == "house":
            print("Dang tai dataset gia nha public va doi cot thanh price...")
            return download_house_public(path)
        if dataset_key == "btc":
            print("Dang tai BTC-USD tu yfinance va doi Close thanh priceUSD...")
            return download_yfinance("BTC-USD", path, "priceUSD")
        if dataset_key == "voltage":
            print("Dang tai Individual Household Electric Power Consumption tu UCI...")
            return download_household_power(path)
        if dataset_key == "nifty":
            print("Dang tai chi so ^NSEI tu yfinance va doi Close thanh close...")
            return download_yfinance("^NSEI", path, "close")
    except Exception as exc:
        print(f"Khong tai duoc dataset public cho {dataset_key}: {exc}")
        print_manual_dataset_instruction(dataset_key)
        if allow_demo:
            print("Tao dataset demo de co the chay thu pipeline RNN.")
            return create_demo_dataset(dataset_key)
        raise

    return load_dataset(dataset_key)


def load_dataset(dataset_key: str, nrows: int | None = None) -> pd.DataFrame:
    """Doc dataset tu thu muc datasets/."""
    config = get_dataset_config(dataset_key)
    path = config["path"]
    if not path.exists():
        raise FileNotFoundError(f"Chua co file {path}. Hay chay script train de tai/tao du lieu.")

    if dataset_key == "voltage":
        return pd.read_csv(path, sep=";", na_values=["?"], nrows=nrows)
    return pd.read_csv(path, nrows=nrows)


def normalize_target_column(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Doi ten cac cot pho bien ve dung ten cot muc tieu cua de bai."""
    df = df.copy()
    candidates = {
        "price": ["price", "Price", "SalePrice", "sale_price", "median_house_value"],
        "priceUSD": ["priceUSD", "Close", "close", "Adj Close", "price_usd"],
        "Voltage": ["Voltage", "voltage"],
        "close": ["close", "Close", "Adj Close"],
    }
    for col in candidates.get(target_col, [target_col]):
        if col in df.columns:
            return df.rename(columns={col: target_col})
    raise ValueError(f"Khong tim thay cot {target_col}. Cac cot hien co: {list(df.columns)}")


def get_target_series(dataset_key: str) -> pd.Series:
    """Lay chuoi gia tri can du bao, da xu ly missing value."""
    config = get_dataset_config(dataset_key)
    df = ensure_dataset(dataset_key)
    df = normalize_target_column(df, config["target"])

    series = pd.to_numeric(df[config["target"]], errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan)
    series = series.ffill().bfill().dropna()
    if len(series) < 30:
        raise ValueError("Dataset qua it dong sau khi xu ly missing value. Can it nhat 30 gia tri.")
    return series.reset_index(drop=True)


def print_manual_dataset_instruction(dataset_key: str) -> None:
    """In huong dan dat dataset thu cong khi khong tai duoc tu mang."""
    config = get_dataset_config(dataset_key)
    instructions = {
        "house": "Tai dataset gia nha public, doi cot gia ban thanh price va luu vao datasets/raw_sales.csv.",
        "btc": "Tai du lieu BTC-USD, doi cot Close thanh priceUSD va luu vao datasets/BTC_DATA.csv.",
        "voltage": "Tai household_power_consumption.txt tu UCI va dat vao datasets/household_power_consumption.txt.",
        "nifty": "Tai NIFTY-50 hoac yfinance ^NSEI, doi cot Close thanh close va luu vao datasets/NIFTY_stock_market.csv.",
    }
    print(instructions[dataset_key])
    print(f"Duong dan can co: {config['path']}")
