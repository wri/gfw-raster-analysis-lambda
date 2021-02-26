from typing import Union
import logging
import os
from shapely.geometry import Polygon, MultiPolygon

from raster_analysis.layer.layer import DataLakeLayerInfoManager

Numeric = Union[int, float]
BasePolygon = Union[Polygon, MultiPolygon]
ResultValue = Union[int, float, str]

LOGGING_LEVEL = logging.INFO
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(LOGGING_LEVEL)

RESULTS_CHECK_INTERVAL = 0.05
RESULTS_CHECK_TRIES = 30 / RESULTS_CHECK_INTERVAL

TILE_WIDTH = 1
WINDOW_SIZE = 4000
GRID_SIZE = 10
GRID_COLS = 40000

S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", None)
LAMBDA_ENDPOINT_URL = os.environ.get("LAMBDA_ENDPOINT_URL", None)
DYNAMODB_ENDPOINT_URL = os.environ.get("DYNAMODB_ENDPOINT_URL", None)
AWS_REGION = "us-east-1"

DATA_LAKE_LAYER_MANAGER = DataLakeLayerInfoManager()

FANOUT_LAMBDA_NAME = os.environ.get("FANOUT_LAMBDA_NAME", "")
RASTER_ANALYSIS_LAMBDA_NAME = os.environ.get("RASTER_ANALYSIS_LAMBDA_NAME", "")

LAMBDA_ASYNC_PAYLOAD_LIMIT_BYTES = 250000
DYMANODB_TTL_SECONDS = 3600

CO2_FACTOR = 0.47 * 44 / 12