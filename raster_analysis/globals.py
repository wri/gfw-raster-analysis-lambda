from typing import Union
import logging
import os

Numeric = Union[int, float]

LOGGING_LEVEL = logging.INFO
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(LOGGING_LEVEL)

RESULTS_CHECK_INTERVAL = 0.05
RESULTS_CHECK_TRIES = 30 / RESULTS_CHECK_INTERVAL

TILE_WIDTH = 1

S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", None)
LAMBDA_ENDPOINT_URL = os.environ.get("LAMBDA_ENDPOINT_URL", None)
DYNAMODB_ENDPOINT_URL = os.environ.get("DYNAMODB_ENDPOINT_URL", None)
AWS_REGION = "us-east-1"
