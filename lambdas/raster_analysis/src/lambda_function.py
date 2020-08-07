import logging
import sys
import traceback
import os

from shapely.geometry import shape

from raster_analysis import geoprocessing
from raster_analysis.boto import dynamodb_resource

from lambda_decorators import json_http_resp
from copy import deepcopy
from aws_xray_sdk.core import xray_recorder
from decimal import Decimal
from datetime import datetime

fmt = "%(asctime)s %(levelname)-4s - %(name)s - %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)


@json_http_resp
@xray_recorder.capture("Raster Analysis Lambda")
def handler(event, context):
    # subsegment.put_annotation("RequestID", context.aws_request_id)
    # subsegment.put_annotation("LogStream", context.log_stream_name)
    # subsegment.put_metadata("RequestParams", event)

    try:
        logger.info(f"Running analysis with parameters: {event}")
        geometry = shape(event["geometry"])

        if "tile" in event:
            tile = shape(event["tile"])
            geometry = geometry.intersection(tile)

        start_date = try_parsing_date(event.get("start_date", None))
        end_date = try_parsing_date(event.get("end_date", None))

        result = geoprocessing.zonal_sql(
            geometry,
            tile,
            event.get("group_by", []),
            event.get("sum", []),
            event.get("filters", []),
            start_date,
            end_date,
        )

        write_result(event, context, result)

        return result
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        raise Exception(f"Internal Server Error <{context.aws_request_id}>")


def write_result(event, context, result):
    if event.get("write_to_dynamo", False):
        dynamo_result = deepcopy(result)

        for layer, col in dynamo_result.items():
            if isinstance(col, float):
                dynamo_result[layer] = Decimal(str(col))
            elif isinstance(col, list) and len(col) > 0 and isinstance(col[0], float):
                dynamo_result[layer] = [Decimal(str(val)) for val in col]

        table = dynamodb_resource().Table(os.environ["TILED_RESULTS_TABLE_NAME"])

        table.put_item(
            Item={
                "analysis_id": event["analysis_id"],
                "tile_id": context.aws_request_id,
                "result": dynamo_result,
            }
        )


def try_parsing_date(text):
    if text:
        for fmt in ("%Y-%m-%d", "%Y"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass
        raise ValueError("no valid date format found")


"""
repro weird issue:

[-55.0, -12.5],
[-56.25, -12.5],
[-56.25, -11.25],
[-55.0, -11.25],
[-55.0, -12.5],

"""
