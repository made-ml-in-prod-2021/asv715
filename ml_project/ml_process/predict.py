import click
from ml_process.data import read_data
from ml_process.models.model_fit_predict import load_model, predict_model
from ml_process.utils.savers import save_predictions
from ml_process.entities.train_pipeline_params import read_training_pipeline_params
from ml_process.features.build_features import make_features
from ml_process.utils.logger import logger


@click.command(name="predict")
@click.argument("config_path")
@click.argument("model_path")
@click.argument("test_sample_path")
def predict_command(config_path: str, model_path: str, test_sample_path: str) -> None:
    logger.info("Start prediction mode")
    params = read_training_pipeline_params(config_path)

    logger.info("Load model from file")
    model = load_model(model_path)

    logger.info("Read test data")
    data = read_data(test_sample_path)

    features, _ = make_features(
        data,
        params.feature_params,
        params.processing_params,
        handle_target=False
    )

    logger.info("Make predictions")
    predictions = predict_model(model, features)

    logger.info("Save predictions info file")
    save_predictions(params.predict_path, predictions)

    logger.info("Finish prediction mode")


if __name__ == "__main__":
    predict_command()
