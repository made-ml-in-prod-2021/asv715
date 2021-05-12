"""
Functions to work with raw data
"""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_process.entities import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    """
    Read dataset from path
    :param path: path to dataset
    :return: dataframe
    """
    return pd.read_csv(path)


def split_train_val_data(
    data: pd.DataFrame,
    params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split full dataset on train and validation parts
    :param data: full dataset
    :param params: splitting params
    :return: tuple with two dataframes
    """
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state, shuffle=params.shuffle
    )

    return train_data, val_data
