from typing import Dict
import json
import numpy as np


def save_metrics(metric_path: str, metrics: Dict[str, float]) -> None:
    """
    Save metrics into file
    :param metric_path: path to metric file
    :param metrics: metrics dictionary
    :return: None
    """
    with open(metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)


def save_predictions(predict_path: str, predicts: np.ndarray) -> None:
    """
    Save predictions into file
    :param predict_path: path to file with predictions
    :param predicts: array with predictions
    :return: None
    """
    with open(predict_path, "w") as predicts_file:
        dumped = json.dumps(predicts.tolist())
        json.dump(dumped, predicts_file)
