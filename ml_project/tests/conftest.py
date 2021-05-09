import os
import pytest
import numpy as np
import pandas as pd
from typing import List, Tuple
from ml_process.features import make_features
from ml_process.entities.feature_params import FeatureParams
from ml_process.entities.processing_params import ProcessingParams

PATH_TO_SAMPLE_DATASET = 'sample.csv'


@pytest.fixture()
def sample_dataset_path():
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, PATH_TO_SAMPLE_DATASET)


@pytest.fixture()
def target_col():
    return "target"


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "sex",
        "fbs",
        "restecg"
    ]


@pytest.fixture()
def numerical_features() -> List[str]:
    return [
        "cp",
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "exang"
    ]


@pytest.fixture()
def fake_data() -> pd.DataFrame:
    total_rows = 10
    data = {
        "sex": np.random.choice([0, 1], total_rows),
        "fbs": np.random.choice([0, 1], total_rows),
        "restecg": np.random.choice([0, 1], total_rows),
        "cp": np.random.choice([0, 1, 2, 3], total_rows),
        "age": np.random.randint(30, 60, total_rows),
        "trestbps": np.random.randint(100, 150, total_rows),
        "chol": np.random.randint(160, 270, total_rows),
        "thalach": np.random.randint(130, 190, total_rows),
        "oldpeak": np.random.random(total_rows) * 5,
        "slope": np.random.choice([0, 1, 2], total_rows),
        "ca": np.random.choice([0, 1, 2], total_rows),
        "thal": np.random.choice([2, 3], total_rows),
        "exang": np.random.choice([0, 1], total_rows),
        "target": np.random.choice([0, 1], total_rows),
    }

    return pd.DataFrame(data)


@pytest.fixture()
def dataset(
    fake_data: pd.DataFrame,
    categorical_features: List[str],
    numerical_features: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    feature_params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col="target"
    )

    processing_params = ProcessingParams(
        use_scaler=False,
        use_poly_features=False
    )

    return make_features(fake_data, feature_params, processing_params)


@pytest.fixture()
def test_features_array() -> np.ndarray:
    arr = np.array([
        [1, 4, 2],
        [2, 6, 7],
        [3, 8, 1]
    ])

    return arr


@pytest.fixture()
def test_features_powered_array(test_features_array: np.ndarray) -> np.ndarray:
    return test_features_array + test_features_array ** 2
