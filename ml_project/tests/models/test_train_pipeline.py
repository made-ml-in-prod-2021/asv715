import pytest
import pickle
import os
from py._path.local import LocalPath
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from ml_process.train_pipeline import train_pipeline
from ml_process.models.model_fit_predict import train_model, serialize_model
from ml_process.entities.train_params import TrainingParams
from ml_process.entities.train_pipeline_params import TrainingPipelineParams
from ml_process.entities.feature_params import FeatureParams
from ml_process.entities.split_params import SplittingParams
from ml_process.entities.processing_params import ProcessingParams


def test_can_train_model(dataset):
    features, target = dataset
    model = train_model(features, target, TrainingParams())

    assert isinstance(model, LogisticRegression), "Model is of wrong instance"
    check_is_fitted(model)


def test_can_throw_exception_for_wrong_model_type(dataset):
    with pytest.raises(ValueError):
        features, target = dataset
        params = TrainingParams(
            model_type="FakeClassifier"
        )
        model = train_model(features, target, params)


def test_serialize_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    model = LogisticRegression()
    real_output = serialize_model(model, expected_output)

    assert real_output == expected_output
    assert os.path.exists

    with open(real_output, "rb") as f:
        model = pickle.load(f)

    assert isinstance(model, LogisticRegression), "Loaded model is of wrong instance"


def test_can_run_full_process(
    tmpdir: LocalPath,
    sample_dataset_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    expected_predict_path = tmpdir.join("predictions.json")

    params = TrainingPipelineParams(
        input_data_path=sample_dataset_path,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        predict_path=expected_predict_path,
        splitting_params=SplittingParams(val_size=0.3, random_state=17),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            use_log_trick=False,
        ),
        processing_params=ProcessingParams(
            use_scaler=True,
            use_poly_features=True,
            poly_features_max_power=2
        ),
        train_params=TrainingParams(model_type="LogisticRegression")
    )

    real_model_path, metrics = train_pipeline(params)

    assert "roc_auc_score" in metrics, "Metrics have no ROC-AUC score"
    assert "accuracy_score" in metrics, "Metrics have no Accuracy score"

    assert metrics["roc_auc_score"] > 0, "ROC-AUC score has invalid value"
    assert metrics["accuracy_score"] > 0, "Accuracy score has invalid value"

    assert os.path.exists(real_model_path), "Model was not serialized"
    assert os.path.exists(params.metric_path), "Metrics were not saved"
    assert os.path.exists(params.predict_path), "Predictions were not saved"
