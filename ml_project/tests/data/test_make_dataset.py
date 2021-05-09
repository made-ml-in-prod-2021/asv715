import pandas as pd
from ml_process.data.make_dataset import read_data, split_train_val_data


def test_can_load_data(sample_dataset_path: str):
    data = read_data(sample_dataset_path)

    assert isinstance(data, pd.DataFrame), "Dataset should be instance of DataFrame"
    assert len(data) > 0, "Dataset should contain data rows"


def test_can_split_data_correctly():
    pass