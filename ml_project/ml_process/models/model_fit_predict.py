"""
Main functions to work with models lifecycle
"""
import pickle
from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from ml_process.entities.train_params import TrainingParams
from ml_process.utils.logger import logger


ClassificationModel = Union[RandomForestClassifier, LogisticRegression, SGDClassifier]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> ClassificationModel:
    """
    Train model in dependence of type
    :param features: features dataframe
    :param target: target column
    :param train_params: train params
    :return: fitted model
    """
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.random_forest_estimators,
            random_state=train_params.random_state,
            max_depth=train_params.max_depth,
            min_samples_split=train_params.min_samples_split,
            min_samples_leaf=train_params.min_samples_leaf
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=train_params.log_reg_max_iterations)
    elif train_params.model_type == "SGDClassifier":
        model = SGDClassifier()
    else:
        logger.critical("Invalid model type")
        raise ValueError()

    model.fit(features, target)

    return model


def predict_model(
    model: ClassificationModel, features: pd.DataFrame, use_log_trick: bool = False
) -> np.ndarray:
    """
    Make prediction by fitted model
    :param model: fitted classification model
    :param features: features dataframe
    :param use_log_trick: whether or not to get exponents of predictions
    :return: numpy array with predictions
    """
    predicts = model.predict(features)

    if use_log_trick:
        predicts = np.exp(predicts)

    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series, use_log_trick: bool = False
) -> Dict[str, float]:
    """
    Calculate metrics
    :param predicts: predictions of the model
    :param target: target column
    :param use_log_trick: whether or not to get exponents of target
    :return: dict with two metrics
    """
    if use_log_trick:
        target = np.exp(target)

    return {
        "accuracy_score": accuracy_score(target, predicts),
        "roc_auc_score": roc_auc_score(target, predicts)
    }


def serialize_model(model: ClassificationModel, output: str) -> str:
    """
    Serialize fitted model into file
    :param model: fitted classification model
    :param output: path to output file
    :return: path to output file
    """
    with open(output, "wb") as model_file:
        pickle.dump(model, model_file)

    return output


def load_model(model_path: str):
    """
    Load model from file
    :param model_path: path to file with model
    :return: model
    """
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    return model
