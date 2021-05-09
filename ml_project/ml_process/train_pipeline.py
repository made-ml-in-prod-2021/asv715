import json
from typing import Tuple, Dict
import click
import pandas as pd
from ml_process.data import read_data, split_train_val_data
from ml_process.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from ml_process.features import make_features
from ml_process.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)
from ml_process.utils.logger import setup_logging, logger
from ml_process.utils.savers import save_predictions, save_metrics


def train_pipeline(training_pipeline_params: TrainingPipelineParams) -> Tuple[str, Dict]:
    """
    Entry point to train, predict and evaluate pipeline
    :param training_pipeline_params:  params for training
    :return: path to serialized model and metrics in json format
    """
    logger.info(f"Start pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"Data shape is {data.shape}")

    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"Train dataframe shape is {train_df.shape}")
    logger.info(f"Validation dataframe shape is {val_df.shape}")

    logger.info("Start data processing")
    train_features, train_target = make_features(
        train_df,
        training_pipeline_params.feature_params,
        training_pipeline_params.processing_params
    )
    val_features, val_target = make_features(
        val_df,
        training_pipeline_params.feature_params,
        training_pipeline_params.processing_params
    )

    logger.info("Start train")
    logger.info(f"Train features shape is {train_features.shape}")

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    logger.info("Prepare features for prediction")
    val_features_prepared = prepare_val_features_for_predict(
        train_features, val_features
    )

    logger.info("Start prediction")
    logger.info(f"Validation features shape is {val_features_prepared.shape}")
    predicts = predict_model(
        model,
        val_features_prepared,
        training_pipeline_params.feature_params.use_log_trick,
    )
    logger.info("Save predictions")
    save_predictions(training_pipeline_params.predict_path, predicts)

    logger.info("Start evaluation")
    metrics = evaluate_model(
        predicts,
        val_target,
        use_log_trick=training_pipeline_params.feature_params.use_log_trick,
    )

    logger.info(f"Metrics is {metrics}")
    logger.info("Save metrics")
    save_metrics(training_pipeline_params.metric_path, metrics)

    logger.info("Serialize model")
    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)

    logger.info("Process finished")

    return path_to_model, metrics


def prepare_val_features_for_predict(
    train_features: pd.DataFrame, val_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Transform features before prediction
    :param train_features: features in train dataset
    :param val_features: features in validation dataset
    :return: validation features
    """
    train_features, val_features = train_features.align(
        val_features, join="left", axis=1
    )
    val_features = val_features.fillna(0)

    return val_features


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str) -> None:
    """
    Start process
    :param config_path: path to main config
    :return: None
    """
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    setup_logging()
    train_pipeline_command()
