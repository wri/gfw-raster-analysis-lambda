import sys
import json

from shapely.geometry import shape, mapping
from aws_xray_sdk.core import patch
from aws_xray_sdk.core import xray_recorder
import shapely.wkt

from raster_analysis.tiling import AnalysisTiler
from raster_analysis.geometry import encode_geometry
from raster_analysis.globals import LOGGER, TILE_WIDTH, LAMBDA_ASYNC_PAYLOAD_LIMIT_BYTES

patch(["boto3"])


@xray_recorder.capture("Tiled Analysis")
def handler(event, context):
    try:
        query = event["query"]
        geojson = event["geometry"]

        tiler = AnalysisTiler(query, geojson, context.aws_request_id)
        tiler.execute()
        results = tiler.result_as_csv().getvalue()

        LOGGER.info("Successfully merged tiled results")
        return {"statusCode": 200, "body": {"status": "success", "data": results}}
    except Exception as e:
        LOGGER.exception(e)
        return {"statusCode": 500, "body": {"status": "failed", "message": str(e)}}
