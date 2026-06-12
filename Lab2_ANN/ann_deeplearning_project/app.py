"""Flask app demo 5 mô hình ANN trong bài thực hành."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from utils.labels import (
    ADULT_DEFAULT_OPTIONS,
    ADULT_FEATURES,
    ADULT_LABELS,
    CAR_FEATURES,
    CAR_LABELS,
    CAR_OPTIONS,
    CATDOG_LABELS,
    CIFAR10_LABELS,
    MNIST_LABELS,
)
from utils.preprocess import MODELS_DIR, UPLOAD_DIR, ensure_project_dirs, preprocess_image_for_model

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
ensure_project_dirs()

MODEL_CACHE = {}


IMAGE_TASKS = {
    "cifar10": {
        "title": "CIFAR10",
        "model": "ann_cifar10.h5",
        "labels": CIFAR10_LABELS,
        "image_size": (32, 32),
        "color_mode": "rgb",
    },
    "mnist": {
        "title": "MNIST",
        "model": "ann_mnist.h5",
        "labels": MNIST_LABELS,
        "image_size": (28, 28),
        "color_mode": "grayscale",
    },
    "catdog": {
        "title": "Cat/Dog",
        "model": "ann_catdog.h5",
        "labels": CATDOG_LABELS,
        "image_size": (64, 64),
        "color_mode": "rgb",
    },
}


def load_keras_model(model_name):
    """Load model Keras khi cần dùng, tránh load tất cả ngay lúc mở web."""
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        return None

    if model_name not in MODEL_CACHE:
        from tensorflow.keras.models import load_model

        MODEL_CACHE[model_name] = load_model(model_path, compile=False)
    return MODEL_CACHE[model_name]


def model_missing_message():
    return "Chưa có model. Vui lòng chạy file train tương ứng trước."


def dense_array(data):
    """Đổi sparse matrix của OneHotEncoder sang dense array nếu cần."""
    return data.toarray() if hasattr(data, "toarray") else data


def predict_image_task(task, file_storage):
    config = IMAGE_TASKS[task]
    model = load_keras_model(config["model"])
    if model is None:
        return {"error": model_missing_message()}

    if not file_storage or file_storage.filename == "":
        return {"error": "Vui lòng chọn một ảnh để upload."}

    filename = secure_filename(file_storage.filename)
    image_path = UPLOAD_DIR / filename
    file_storage.save(image_path)

    x = preprocess_image_for_model(
        image_path,
        config["image_size"],
        color_mode=config["color_mode"],
    )
    prediction = model.predict(x, verbose=0)

    if task == "catdog":
        dog_prob = float(prediction[0][0])
        index = 1 if dog_prob >= 0.5 else 0
        confidence = dog_prob if index == 1 else 1.0 - dog_prob
    else:
        index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

    return {
        "task_title": config["title"],
        "label": config["labels"][index],
        "confidence": f"{confidence * 100:.2f}%",
        "image_url": url_for("static", filename=f"uploads/{filename}"),
    }


def predict_adult_income(form):
    model = load_keras_model("ann_adult.h5")
    preprocessor_path = MODELS_DIR / "adult_preprocessor.joblib"
    if model is None or not preprocessor_path.exists():
        return {"error": model_missing_message()}

    numeric_defaults = {
        "age": 30,
        "fnlwgt": 200000,
        "education-num": 10,
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
    }

    row = {}
    for feature in ADULT_FEATURES:
        value = form.get(feature, "")
        if feature in numeric_defaults:
            row[feature] = float(value or numeric_defaults[feature])
        else:
            row[feature] = value

    preprocessor = joblib.load(preprocessor_path)
    x = dense_array(preprocessor.transform(pd.DataFrame([row])))
    probability = float(model.predict(x, verbose=0)[0][0])
    label = ADULT_LABELS[1] if probability >= 0.5 else ADULT_LABELS[0]
    confidence = probability if probability >= 0.5 else 1.0 - probability

    return {
        "task_title": "Adult Income",
        "label": label,
        "confidence": f"{confidence * 100:.2f}%",
    }


def predict_car_evaluation(form):
    model = load_keras_model("ann_car.h5")
    preprocessor_path = MODELS_DIR / "car_preprocessor.joblib"
    label_encoder_path = MODELS_DIR / "car_label_encoder.joblib"
    if model is None or not preprocessor_path.exists() or not label_encoder_path.exists():
        return {"error": model_missing_message()}

    row = {feature: form.get(feature, CAR_OPTIONS[feature][0]) for feature in CAR_FEATURES}
    preprocessor = joblib.load(preprocessor_path)
    label_encoder = joblib.load(label_encoder_path)
    x = dense_array(preprocessor.transform(pd.DataFrame([row])))
    prediction = model.predict(x, verbose=0)
    index = int(np.argmax(prediction[0]))
    label = label_encoder.inverse_transform([index])[0]

    return {
        "task_title": "Car Evaluation",
        "label": label,
        "confidence": f"{float(np.max(prediction[0])) * 100:.2f}%",
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        task = request.form.get("task")

        if task in IMAGE_TASKS:
            result = predict_image_task(task, request.files.get("image"))
        elif task == "adult":
            result = predict_adult_income(request.form)
        elif task == "car":
            result = predict_car_evaluation(request.form)
        else:
            result = {"error": "Vui lòng chọn bài toán cần dự đoán."}

        return render_template("result.html", result=result)

    return render_template(
        "index.html",
        adult_options=ADULT_DEFAULT_OPTIONS,
        car_options=CAR_OPTIONS,
    )


if __name__ == "__main__":
    app.run(debug=True)
