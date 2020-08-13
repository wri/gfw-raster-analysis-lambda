import logging
import sys
import traceback

from shapely.geometry import shape
from raster_analysis import geoprocessing
from raster_analysis.results_store import AnalysisResultsStore

from lambda_decorators import json_http_resp
from aws_xray_sdk.core import xray_recorder
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

        results_store = AnalysisResultsStore(event["analysis_id"])
        results_store.save_result(result, context.aws_request_id)

        return result
    except Exception as e:
        logger.exception(e)
        raise Exception(f"Internal Server Error <{context.aws_request_id}>")


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
