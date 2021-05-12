"""
Dataclass and function for train params
"""
from dataclasses import dataclass
import yaml
from marshmallow_dataclass import class_schema
from .split_params import SplittingParams
from .feature_params import FeatureParams
from .processing_params import ProcessingParams
from .train_params import TrainingParams


# pylint: disable=too-many-instance-attributes
@dataclass()
class TrainingPipelineParams:
    """
    Dataclass for train params
    """
    input_data_path: str
    output_model_path: str
    metric_path: str
    predict_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    processing_params: ProcessingParams
    train_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    """
    Load params from config in yml format
    :param path: path to config
    :return: params schema
    """
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()

        return schema.load(yaml.safe_load(input_stream))
