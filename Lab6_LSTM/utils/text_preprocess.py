"""Ham xu ly van ban va sinh cau bang LSTM."""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


def clean_vietnamese_text(text):
    """Lam sach van ban tieng Viet, giu lai chu co dau va dau cau co ban."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s\.\,\!\?\n]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_twitter_text(text):
    """Lam sach tweet: bo URL, mention, ky tu # nhung giu noi dung hashtag."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", "")
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s\.\!\?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_text_lines(path, min_words=2):
    """Doc file text va tra ve cac dong da lam sach."""
    path = Path(path)
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        clean_line = clean_vietnamese_text(line)
        if len(clean_line.split()) >= min_words:
            lines.append(clean_line)
    return lines


def read_twitter_lines(csv_path, min_words=3, max_rows=50000):
    """Doc file CSV Twitter va chuan hoa cot noi dung thanh twitter_content."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path, encoding_errors="ignore")
    possible_columns = ["twitter_content", "text", "content", "tweet", "selected_text"]
    content_col = next((col for col in possible_columns if col in df.columns), None)
    if content_col is None:
        raise ValueError(
            "CSV can co cot twitter_content, text, content, tweet hoac selected_text"
        )
    if content_col != "twitter_content":
        df = df.rename(columns={content_col: "twitter_content"})

    lines = []
    for text in df["twitter_content"].dropna().astype(str).head(max_rows):
        clean_line = clean_twitter_text(text)
        if len(clean_line.split()) >= min_words:
            lines.append(clean_line)
    return lines


def create_ngram_dataset(lines, max_num_words=10000):
    """Tao tap n-gram de du doan tu tiep theo."""
    tokenizer = Tokenizer(num_words=max_num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(lines)

    input_sequences = []
    for line in lines:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(2, len(token_list) + 1):
            input_sequences.append(token_list[:i])

    if not input_sequences:
        raise ValueError("Khong du du lieu de tao n-gram sequence")

    max_sequence_len = max(len(seq) for seq in input_sequences)
    input_sequences = pad_sequences(
        input_sequences,
        maxlen=max_sequence_len,
        padding="pre",
    )
    xs = input_sequences[:, :-1]
    labels = input_sequences[:, -1]
    total_words = min(max_num_words, len(tokenizer.word_index) + 1)
    ys = to_categorical(labels, num_classes=total_words)
    return xs, ys, tokenizer, total_words, max_sequence_len


def save_tokenizer(tokenizer, max_sequence_len, path):
    """Luu tokenizer kem max_sequence_len de luc du doan co the padding dung."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"tokenizer": tokenizer, "max_sequence_len": max_sequence_len}, path)


def load_tokenizer_bundle(path):
    """Doc tokenizer va max_sequence_len tu file joblib/pkl."""
    bundle = joblib.load(path)
    if isinstance(bundle, dict):
        return bundle["tokenizer"], bundle["max_sequence_len"]
    return bundle, getattr(bundle, "max_sequence_len", None)


def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len, end_with_period=False):
    """Sinh van ban bang cach lap lai viec du doan tu tiep theo."""
    text = clean_vietnamese_text(seed_text)
    index_word = {idx: word for word, idx in tokenizer.word_index.items()}
    next_words = int(next_words)

    for _ in range(max(1, next_words)):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_id = int(np.argmax(predicted_probs))
        output_word = index_word.get(predicted_id, "")
        if not output_word or output_word == "<OOV>":
            break
        text += " " + output_word
        if end_with_period and output_word.endswith((".", "!", "?")):
            break

    text = re.sub(r"\s+", " ", text).strip()
    if end_with_period and text and text[-1] not in ".!?":
        text += "."
    return text


def build_text_lstm_model(total_words, max_sequence_len, lstm_units=150):
    """Tao model LSTM sinh van ban."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len - 1),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(total_words, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

