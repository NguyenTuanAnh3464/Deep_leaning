from flask import Flask, render_template, request

from utils.data_loader import DATASET_CONFIG
from utils.prediction import MissingModelError, parse_recent_values, predict_next_value


app = Flask(__name__)

MODEL_OPTIONS = {
    "house": "Dự báo giá nhà",
    "btc": "Dự báo giá Bitcoin",
    "voltage": "Dự báo Voltage",
    "nifty": "Dự báo NIFTY close",
}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", options=MODEL_OPTIONS)


@app.route("/predict", methods=["POST"])
def predict():
    dataset_key = request.form.get("model_key", "house")
    raw_values = request.form.get("recent_values", "")
    time_steps = int(request.form.get("time_steps", 12))

    try:
        values = parse_recent_values(raw_values)
        prediction = predict_next_value(dataset_key, values, time_steps=time_steps)
        config = DATASET_CONFIG[dataset_key]
        return render_template(
            "result.html",
            options=MODEL_OPTIONS,
            selected_key=dataset_key,
            prediction=prediction,
            target=config["target"],
            error=None,
            recent_values=raw_values,
            time_steps=time_steps,
        )
    except MissingModelError as exc:
        error = str(exc)
    except ValueError as exc:
        error = f"Dữ liệu nhập chưa hợp lệ: {exc}"
    except Exception as exc:
        error = f"Có lỗi khi dự báo: {exc}"

    return render_template(
        "result.html",
        options=MODEL_OPTIONS,
        selected_key=dataset_key,
        prediction=None,
        target=DATASET_CONFIG.get(dataset_key, DATASET_CONFIG["house"])["target"],
        error=error,
        recent_values=raw_values,
        time_steps=time_steps,
    )


if __name__ == "__main__":
    app.run(debug=True)
