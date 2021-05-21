"""
Extract and transform features and target from the dataframe
"""
from typing import Tuple
import numpy as np
import pandas as pd
from ml_process.entities.feature_params import FeatureParams
from ml_process.entities.processing_params import ProcessingParams
from .transformer import FeaturesTransformer


def make_features(
    dataframe: pd.DataFrame,
    feature_params: FeatureParams,
    processing_params: ProcessingParams,
    handle_target: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Transform initial dataframe and extract features and target
    :param dataframe: initial dataframe
    :param feature_params: feature params
    :param processing_params: params for processing feature types and values
    :param handle_target: need to extract target
    :return: transformed dataframe and target in Series
    """
    transformer = FeaturesTransformer(feature_params, processing_params)

    if handle_target:
        features = dataframe.drop([feature_params.target_col], axis=1)
        target = dataframe[feature_params.target_col]

        if feature_params.use_log_trick:
            target = pd.Series(np.log(target.to_numpy()))

        features = transformer.transform(features)
    else:
        target = None
        features = dataframe.copy()

    return pd.DataFrame(features), target
