from flask import Flask, request, jsonify, render_template
import os
import sys
import json
import pickle

from src.exception import CustomException
from src.utils import load_object
from src.pipeline.predict_pipline import PredictPipeline, CustomData
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

app = Flask(__name__)

ARTIFACTS_DIR = "artifacts"

# ──────────────────────────────────────────
# Helper: load model summary from artifacts
# ──────────────────────────────────────────
def load_model_summary():
    summary_path = os.path.join(ARTIFACTS_DIR, "model_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    return []


# ──────────────────────────────────────────
# Routes
# ──────────────────────────────────────────

@app.route("/")
def index():
    summary = load_model_summary()
    return render_template("index.html", summary=summary)


@app.route("/info")
def info_page():
    return render_template("model_info.html")


@app.route("/predict_page")
def predict_page():
    summary = load_model_summary()
    model_names = [s["model"] for s in summary]
    return render_template("predict.html", model_names=model_names)


@app.route("/train", methods=["GET"])
def train():
    try:
        # Data Ingestion
        ingestion_obj = DataIngestion()
        train_data, test_data = ingestion_obj.initiate_data_ingestion()

        # Data Transformation
        transformation_obj = DataTransformation()
        train_arr, test_arr, _ = transformation_obj.intiate_dataTraformation(train_data, test_data)

        # Model Training
        trainer_obj = ModelTrainer()
        summary = trainer_obj.intiate_model_training(train_arr, test_arr)

        # Save summary to disk
        summary_path = os.path.join(ARTIFACTS_DIR, "model_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f)

        return jsonify({"status": "success", "summary": summary})

    except Exception as e:
        raise CustomException(e, sys)


@app.route("/models", methods=["GET"])
def get_models():
    try:
        summary = load_model_summary()
        if not summary:
            return jsonify({"status": "error", "message": "No trained models found. Please train first."}), 404
        return jsonify({"status": "success", "models": summary})
    except Exception as e:
        raise CustomException(e, sys)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        model_name = data.pop("model_name")

        custom_data = CustomData(**data)
        df = custom_data.get_data_as_dataframe()

        pipeline = PredictPipeline()
        prediction = pipeline.predict(model_name=model_name, features=df)

        return jsonify({
            "status": "success",
            "model_used": model_name,
            "prediction": prediction[0]
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
