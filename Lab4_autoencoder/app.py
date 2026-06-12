"""Flask app nhan dang anh bang Autoencoder + Encoder classifier."""

from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from utils.labels import CATDOG_LABELS, CIFAR10_LABELS, FASHION_MNIST_LABELS, GENDER_LABELS
from utils.preprocess import MODELS_DIR, UPLOAD_DIR, ensure_project_dirs, preprocess_image_for_classifier


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
ensure_project_dirs()

MODEL_CACHE = {}

TASKS = {
    "cifar10": {
        "title": "CIFAR10",
        "classifier": "ae_cifar10_classifier.h5",
        "target_size": (32, 32),
        "grayscale": False,
        "labels": CIFAR10_LABELS,
    },
    "catdog": {
        "title": "Cat/Dog",
        "classifier": "ae_catdog_classifier.h5",
        "target_size": (64, 64),
        "grayscale": False,
        "labels": CATDOG_LABELS,
    },
    "fashion": {
        "title": "Fashion-MNIST",
        "classifier": "ae_fashion_classifier.h5",
        "target_size": (28, 28),
        "grayscale": True,
        "labels": FASHION_MNIST_LABELS,
    },
    "gender": {
        "title": "Nam/Nữ",
        "classifier": "ae_gender_classifier.h5",
        "target_size": (64, 64),
        "grayscale": False,
        "labels": GENDER_LABELS,
    },
}


def model_missing_message():
    return "Chưa có model. Vui lòng chạy file train tương ứng trước."


def load_classifier(model_name):
    """Load model khi can dung, tranh load tat ca luc mo web."""
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        return None

    if model_name not in MODEL_CACHE:
        from tensorflow.keras.models import load_model

        MODEL_CACHE[model_name] = load_model(model_path, compile=False)
    return MODEL_CACHE[model_name]


def allowed_file(filename):
    return Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def predict_uploaded_image(task_key, file_storage):
    if task_key not in TASKS:
        return {"error": "Vui lòng chọn mô hình cần dự đoán."}

    if not file_storage or file_storage.filename == "":
        return {"error": "Vui lòng upload một ảnh."}

    if not allowed_file(file_storage.filename):
        return {"error": "File upload phải là ảnh jpg, jpeg, png, bmp hoặc webp."}

    config = TASKS[task_key]
    model = load_classifier(config["classifier"])
    if model is None:
        return {"error": model_missing_message()}

    filename = secure_filename(file_storage.filename)
    image_path = UPLOAD_DIR / filename
    file_storage.save(image_path)

    x = preprocess_image_for_classifier(
        image_path,
        config["target_size"],
        grayscale=config["grayscale"],
    )
    prediction = model.predict(x, verbose=0)

    labels = config["labels"]
    if len(labels) == 2 and prediction.shape[-1] == 1:
        positive_probability = float(prediction[0][0])
        index = 1 if positive_probability >= 0.5 else 0
        confidence = positive_probability if index == 1 else 1.0 - positive_probability
    else:
        index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

    return {
        "task_title": config["title"],
        "label": labels[index],
        "confidence": f"{confidence * 100:.2f}%",
        "image_url": url_for("static", filename=f"uploads/{filename}"),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        task_key = request.form.get("task", "")
        result = predict_uploaded_image(task_key, request.files.get("image"))
        return render_template("result.html", result=result, tasks=TASKS)

    return render_template("index.html", tasks=TASKS)


if __name__ == "__main__":
    app.run(debug=True)

