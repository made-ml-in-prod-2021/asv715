import pytest
from app import inference


@pytest.fixture
def client():
    inference.app.config["TESTING"] = True

    with inference.app.test_client() as client:
        yield client


@pytest.fixture
def fake_valid_data():
    features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "oldpeak2", "slope2", "ca2", "thal2"
    ]
    data = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 2.3, 0, 0, 1]

    return data, features


@pytest.fixture
def fake_invalid_data():
    data = [63, 1, 3, 145, 233, 1, 0, 2.3, 0, 0, 1, 2.3, 0, 0, 1]
    features = [
        "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "oldpeak2", "slope2", "ca2", "thal2"
    ]

    return data, features
