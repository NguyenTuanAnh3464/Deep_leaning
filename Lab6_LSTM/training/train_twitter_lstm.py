import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from utils.image_preprocess import ensure_project_dirs
from utils.text_preprocess import (
    build_text_lstm_model,
    create_ngram_dataset,
    generate_text,
    read_twitter_lines,
    save_tokenizer,
)

EPOCHS = 3
CSV_PATH = PROJECT_ROOT / "datasets" / "twitter_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_twitter.h5"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizers" / "tokenizer_twitter.pkl"


def main():
    ensure_project_dirs()
    lines = read_twitter_lines(CSV_PATH, min_words=3)
    if not lines:
        print("Chua co du lieu Twitter.")
        print("Hay tai Sentiment140 hoac Twitter Sentiment Analysis Dataset.")
        print("Sau do dat file vao datasets/twitter_data.csv voi cot twitter_content.")
        return

    print("Mot vai dong du lieu Twitter:")
    for line in lines[:5]:
        print("-", line)

    xs, ys, tokenizer, total_words, max_sequence_len = create_ngram_dataset(lines)
    model = build_text_lstm_model(total_words, max_sequence_len, lstm_units=150)
    model.summary()
    history = model.fit(xs, ys, epochs=EPOCHS, verbose=1)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    save_tokenizer(tokenizer, max_sequence_len, TOKENIZER_PATH)
    print(f"Da luu model vao: {MODEL_PATH}")
    print(f"Da luu tokenizer vao: {TOKENIZER_PATH}")

    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="loss")
    plt.title("Loss - LSTM Twitter")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "static" / "twitter_loss.png")
    plt.show()

    sample = generate_text(
        "today is",
        20,
        model,
        tokenizer,
        max_sequence_len,
        end_with_period=True,
    )
    print("Tweet sinh thu:")
    print(sample)


if __name__ == "__main__":
    main()

