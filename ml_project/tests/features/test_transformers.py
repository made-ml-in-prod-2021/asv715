import numpy as np
from ml_process.features.transformer import PolynomialFeatures


def test_can_apply_poly_transformer_correctly(test_features_array: np.ndarray, test_features_powered_array: np.ndarray):
    transformer = PolynomialFeatures(2)
    result = transformer.transform(test_features_array)

    assert np.allclose(result, test_features_powered_array), "Polynomial transformer failed"
