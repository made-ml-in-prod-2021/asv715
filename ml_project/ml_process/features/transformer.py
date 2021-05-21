"""
Classes for features processing
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ml_process.entities.feature_params import FeatureParams
from ml_process.entities.processing_params import ProcessingParams


class PolynomialFeatures(BaseEstimator, TransformerMixin):
    """
    Create polynomial features
    """
    def __init__(self, power):
        self.power = power

    # pylint: disable=invalid-name
    # pylint: disable=unused-argument
    def fit(self, X, y=None):
        """
        Implement fit method from the base class
        :return: link to class instance
        """
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Implement transform method from the base class
        :param features: features array
        :return: transformed features array
        """
        result = np.zeros_like(features)

        for power in range(1, self.power + 1):
            result += features ** power

        return result


class FeaturesTransformer:
    """
    Features pipeline transformations
    """
    def __init__(
        self, feature_params: FeatureParams, processing_params: ProcessingParams
    ) -> None:
        self.feature_params = feature_params
        self.processing_params = processing_params

    # pylint: disable=no-self-use
    def build_categorical_pipeline(self) -> Pipeline:
        """
        Build pipeline for categorical features
        :return: features pipeline
        """
        return Pipeline(
            [
                ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
                ("one_hot_encoding", OneHotEncoder()),
            ]
        )

    def build_numerical_pipeline(self) -> Pipeline:
        """
        Build pipeline for numerical features
        :return: features pipeline
        """
        pipe = [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ]

        if self.processing_params.use_scaler:
            pipe.append(
                ("scale", StandardScaler())
            )

        if self.processing_params.use_poly_features:
            pipe.append(
                ("poly", PolynomialFeatures(self.processing_params.poly_features_max_power))
            )

        return Pipeline(pipe)

    def transform(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Transform passed dataframe
        :param dataframe: initial features
        :return: array with transformed features
        """
        pipe = []

        if self.feature_params.categorical_features is not None\
           and len(self.feature_params.categorical_features) > 0:
            pipe.append(
                (
                    "categorical_pipeline",
                    self.build_categorical_pipeline(),
                    self.feature_params.categorical_features,
                )
            )

        if self.feature_params.numerical_features is not None \
           and len(self.feature_params.numerical_features) > 0:
            pipe.append(
                (
                    "numerical_pipeline",
                    self.build_numerical_pipeline(),
                    self.feature_params.numerical_features,
                )
            )

        transformer = ColumnTransformer(pipe)

        return transformer.fit_transform(dataframe)
