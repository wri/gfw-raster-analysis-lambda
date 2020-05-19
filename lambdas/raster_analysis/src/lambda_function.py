import logging
import sys
import traceback
import boto3
import os

from shapely.geometry import shape

from raster_analysis import geoprocessing

from lambda_decorators import json_http_resp
from copy import deepcopy
from aws_xray_sdk.core import xray_recorder
from decimal import *

fmt = "%(asctime)s %(levelname)-4s - %(name)s - %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)


@json_http_resp
def handler(event, context):
    with xray_recorder.capture("Raster Analysis Lambda") as subsegment:
        subsegment.put_annotation("RequestID", context.aws_request_id)
        subsegment.put_annotation("LogStream", context.log_stream_name)
        subsegment.put_metadata("RequestParams", event)

        try:
            logger.info(f"Running analysis with parameters: {event}")

            analyses = event["analyses"] if "analyses" in event else ["count", "area"]
            geometry = shape(event["geometry"])

            if "tile" in event:
                tile = shape(event["tile"])
                geometry = geometry.intersection(tile)

            analysis_raster_id = (
                event["analysis_raster_id"] if "analysis_raster_id" in event else []
            )
            contextual_raster_ids = (
                event["contextual_raster_ids"]
                if "contextual_raster_ids" in event
                else []
            )
            aggregate_raster_ids = (
                event["aggregate_raster_ids"] if "aggregate_raster_ids" in event else []
            )
            extent_year = event["extent_year"] if "extent_year" in event else None
            threshold = event["threshold"] if "threshold" in event else None
            start = event["start"] if "start" in event else None
            end = event["end"] if "end" in event else None

            result = geoprocessing.analysis(
                geometry,
                analyses,
                analysis_raster_id,
                contextual_raster_ids,
                aggregate_raster_ids,
                start,
                end,
                extent_year,
                threshold,
            )

            dynamo_result = deepcopy(result)

            for layer, col in dynamo_result.items():
                if len(col) > 0 and isinstance(col[0], float):
                    dynamo_result[layer] = [Decimal(str(val)) for val in col]

            if event.get("write_to_dynamo", False):
                dynamo = boto3.resource("dynamodb")
                table = dynamo.Table(os.environ["TILED_RESULTS_TABLE_NAME"])

                table.put_item(
                    Item={
                        "analysis_id": event["dynamo_id"],
                        "tile_id": context.aws_request_id,
                        "result": dynamo_result,
                    }
                )

            return result
        except Exception:
            logging.error(traceback.format_exc())
            raise Exception(f"Internal Server Error <{context.aws_request_id}>")


"""
repro weird issue:

[-55.0, -12.5],
[-56.25, -12.5],
[-56.25, -11.25],
[-55.0, -11.25],
[-55.0, -12.5],

"""
