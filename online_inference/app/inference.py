"""
Flask application to handle predict endpoint
"""
import os
import pickle
from typing import List, Union, Optional, Dict
from flask import Flask, jsonify, abort, request
from pydantic import BaseModel, conlist
from sklearn.pipeline import Pipeline
import pandas as pd

app = Flask(__name__)

model: Optional[Pipeline] = None
EXPECTED_COLUMNS: List = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "oldpeak2", "slope2", "ca2", "thal2"
]


class Model(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=80, max_items=80)]
    features: List[str]


def make_predict(
    data: List,
    features: List[str]
) -> List:
    """
    Get model predictions for requested data
    :param data: features values
    :param features: features names
    :return: prediction
    """
    global model
    data = pd.DataFrame([data], columns=features)
    predicts = model.predict(data)

    return predicts.tolist()


def check_data(data: List, features: List[str]) -> bool:
    """
    Validate data
    :param data: features values
    :param features: features names
    :return: True if data is valid, False otherwise
    """
    if EXPECTED_COLUMNS != features or len(data) != len(EXPECTED_COLUMNS):
        return False

    return True


@app.before_first_request
def load_model() -> None:
    """
    Load model from file
    :return: None
    """
    global model

    if app.testing:
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, "models/model.pkl")
    else:
        model_path = os.getenv("PATH_TO_MODEL")

    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        raise RuntimeError(err)

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)


@app.route("/predict")
def get_predict() -> Dict:
    """
    Handle predict endpoint
    :return: predictions
    """
    if request.json is None:
        abort(400)

    data = request.json["data"]
    features = request.json["features"]

    if not check_data(data, features):
        abort(400)

    return jsonify(make_predict(data, features))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
