import logging
from lambda_decorators import json_http_resp

from aws_xray_sdk.core import patch
from aws_xray_sdk.core import xray_recorder

from copy import deepcopy
import boto3
import os

patch(["boto3"])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@json_http_resp
@xray_recorder.capture("Tiled Analysis")
def handler(event, context):
    tiles = event.get("tiles", [])
    payload_base = event["payload"]

    lambda_client = boto3.Session().client("lambda")
    raster_analysis_lambda = os.environ["RASTER_ANALYSIS_LAMBDA_NAME"]
    for tile in tiles:
        payload = deepcopy(payload_base)
        payload["tile"] = tile

        try:
            from raster_analysis.tiling import invoke_lambda

            invoke_lambda(payload, raster_analysis_lambda, lambda_client)
        except Exception as e:
            logger.error(
                f"Invoke raster analysis lambda failed for aws request id: {context.aws_request_id}, tile: {tile}"
            )
            raise e
