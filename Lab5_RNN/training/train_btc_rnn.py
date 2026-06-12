import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from utils.preprocessing import train_rnn_for_dataset


TIME_STEPS = 12
EPOCHS = 20


if __name__ == "__main__":
    result = train_rnn_for_dataset("btc", time_steps=TIME_STEPS, epochs=EPOCHS)
    print("Da train xong RNN du bao gia Bitcoin.")
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"MAE : {result['mae']:.4f}")
    print(f"Model da luu: {result['model_path']}")
