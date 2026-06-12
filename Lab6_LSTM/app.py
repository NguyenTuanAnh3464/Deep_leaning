from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from utils.image_preprocess import ensure_project_dirs
from utils.labels import MODEL_PATHS
from utils.prediction import predict_image_lstm, predict_text_lstm

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

IMAGE_TASKS = {
    "cifar10": "Nhận dạng CIFAR10",
    "catdog": "Nhận dạng Cat/Dog",
    "fashion_mnist": "Nhận dạng Fashion-MNIST",
    "gender": "Nhận dạng Nam/Nữ",
}

TEXT_TASKS = {
    "truyen_kieu": "Sinh văn bản Truyện Kiều",
    "twitter": "Sinh văn bản Twitter",
}

TOKENIZER_PATHS = {
    "truyen_kieu": "tokenizers/tokenizer_truyen_kieu.pkl",
    "twitter": "tokenizers/tokenizer_twitter.pkl",
}


def project_path(relative_path):
    return BASE_DIR / relative_path


@app.route("/", methods=["GET", "POST"])
def index():
    ensure_project_dirs()
    message = None
    selected_task = request.form.get("task", "cifar10")

    if request.method == "POST":
        if selected_task in IMAGE_TASKS:
            image_file = request.files.get("image")
            if not image_file or not image_file.filename:
                message = "Vui lòng upload một ảnh để dự đoán."
            else:
                filename = secure_filename(image_file.filename)
                save_path = UPLOAD_DIR / filename
                image_file.save(save_path)
                model_path = project_path(MODEL_PATHS[selected_task])
                try:
                    result = predict_image_lstm(
                        model_path,
                        save_path,
                        selected_task,
                        show_image=False,
                    )
                    return render_template(
                        "image_result.html",
                        task_name=IMAGE_TASKS[selected_task],
                        label=result["label"],
                        confidence=f"{result['confidence']:.2%}",
                        image_path=f"uploads/{filename}",
                    )
                except FileNotFoundError:
                    message = "Chưa có model. Vui lòng chạy file train tương ứng trước."
                except Exception as exc:
                    message = f"Lỗi khi dự đoán ảnh: {exc}"

        elif selected_task in TEXT_TASKS:
            seed_text = request.form.get("seed_text", "").strip()
            next_words = request.form.get("next_words", "20")
            if not seed_text:
                message = "Vui lòng nhập seed_text."
            else:
                model_path = project_path(MODEL_PATHS[selected_task])
                tokenizer_path = project_path(TOKENIZER_PATHS[selected_task])
                try:
                    generated_text = predict_text_lstm(
                        model_path,
                        tokenizer_path,
                        seed_text,
                        next_words,
                        end_with_period=(selected_task == "twitter"),
                    )
                    return render_template(
                        "text_result.html",
                        task_name=TEXT_TASKS[selected_task],
                        seed_text=seed_text,
                        generated_text=generated_text,
                    )
                except FileNotFoundError:
                    message = "Chưa có model. Vui lòng chạy file train tương ứng trước."
                except Exception as exc:
                    message = f"Lỗi khi sinh văn bản: {exc}"

    return render_template(
        "index.html",
        image_tasks=IMAGE_TASKS,
        text_tasks=TEXT_TASKS,
        selected_task=selected_task,
        message=message,
    )


if __name__ == "__main__":
    ensure_project_dirs()
    app.run(debug=True)

