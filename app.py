from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "iris_model.pkl"

app = Flask(__name__)

artifact = joblib.load(MODEL_PATH)

if isinstance(artifact, dict):
    model = artifact["model"]
    target_names = artifact.get("target_names", ["setosa", "versicolor", "virginica"])
else:
    model = artifact
    target_names = ["setosa", "versicolor", "virginica"]


@app.get("/")
def home():
    return render_template("index.html", result=None)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


def _predict_species(sepal_length, sepal_width, petal_length, petal_width):
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]], dtype=float)
    pred_class = int(model.predict(X)[0])
    pred_label = str(target_names[pred_class])
    return pred_class, pred_label


@app.post("/predict")
def predict_form():
    try:
        sl = float(request.form["sepal_length"])
        sw = float(request.form["sepal_width"])
        pl = float(request.form["petal_length"])
        pw = float(request.form["petal_width"])
        _, label = _predict_species(sl, sw, pl, pw)
        return render_template("index.html", result=label)
    except Exception as e:
        return render_template("index.html", result=f"Error: {e}")


@app.post("/predict_json")
def predict_json():
    data = request.get_json(force=True)

    required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        sl = float(data["sepal_length"])
        sw = float(data["sepal_width"])
        pl = float(data["petal_length"])
        pw = float(data["petal_width"])
    except ValueError:
        return jsonify({"error": "All fields must be numeric."}), 400

    pred_class, pred_label = _predict_species(sl, sw, pl, pw)
    return jsonify({"class_id": pred_class, "prediction": pred_label})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)