"""Ham du doan dung chung cho Flask va notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from utils.image_preprocess import load_image_as_sequence
from utils.labels import BINARY_LABELS, CIFAR10_LABELS, FASHION_MNIST_LABELS
from utils.text_preprocess import generate_text, load_tokenizer_bundle


def predict_image_lstm(model_path, image_path, model_type, show_image=True):
    """Load model, xu ly anh, du doan nhan va do tin cay."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("Chua co model. Vui long chay file train tuong ung truoc.")

    model = load_model(model_path)
    sequence, display_image = load_image_as_sequence(image_path, model_type)
    prediction = model.predict(sequence, verbose=0)

    if model_type == "cifar10":
        class_id = int(np.argmax(prediction[0]))
        label = CIFAR10_LABELS[class_id]
        confidence = float(np.max(prediction[0]))
    elif model_type == "fashion_mnist":
        class_id = int(np.argmax(prediction[0]))
        label = FASHION_MNIST_LABELS[class_id]
        confidence = float(np.max(prediction[0]))
    elif model_type in {"catdog", "gender"}:
        score = float(prediction[0][0])
        class_id = 1 if score >= 0.5 else 0
        label = BINARY_LABELS[model_type][class_id]
        confidence = score if class_id == 1 else 1.0 - score
    else:
        raise ValueError("model_type khong hop le")

    print(f"Nhãn dự đoán: {label}")
    print(f"Độ tin cậy: {confidence:.2%}")

    if show_image:
        plt.imshow(display_image, cmap="gray" if model_type == "fashion_mnist" else None)
        plt.title(f"{label} - {confidence:.2%}")
        plt.axis("off")
        plt.show()

    return {"label": label, "confidence": confidence}


def predict_text_lstm(model_path, tokenizer_path, seed_text, next_words, end_with_period=False):
    """Load model/tokenizer va sinh van ban."""
    model_path = Path(model_path)
    tokenizer_path = Path(tokenizer_path)
    if not model_path.exists():
        raise FileNotFoundError("Chua co model. Vui long chay file train tuong ung truoc.")
    if not tokenizer_path.exists():
        raise FileNotFoundError("Chua co tokenizer. Vui long chay file train tuong ung truoc.")

    model = load_model(model_path)
    tokenizer, max_sequence_len = load_tokenizer_bundle(tokenizer_path)
    return generate_text(
        seed_text,
        next_words,
        model,
        tokenizer,
        max_sequence_len,
        end_with_period=end_with_period,
    )

