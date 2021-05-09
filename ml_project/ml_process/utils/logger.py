import logging
import logging.config
import yaml

APPLICATION_NAME = "ml_project"
DEFAULT_LOGGING_CONFIG_FILEPATH = 'configs/logging.conf.yml'

logger = logging.getLogger(APPLICATION_NAME)


def setup_logging() -> None:
    """
    Setup for logging
    :return: None
    """
    with open(DEFAULT_LOGGING_CONFIG_FILEPATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))
