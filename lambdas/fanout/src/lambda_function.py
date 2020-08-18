from aws_xray_sdk.core import patch
from aws_xray_sdk.core import xray_recorder

from copy import deepcopy
import os

from raster_analysis.boto import lambda_client
from raster_analysis.globals import LOGGER

patch(["boto3"])


@xray_recorder.capture("Fanout")
def handler(event, context):
    tiles = event.get("tiles", [])
    payload_base = event["payload"]

    raster_analysis_lambda = os.environ["RASTER_ANALYSIS_LAMBDA_NAME"]
    for tile in tiles:
        payload = deepcopy(payload_base)
        payload["tile"] = tile

        try:
            from raster_analysis.boto import invoke_lambda

            invoke_lambda(payload, raster_analysis_lambda, lambda_client())
        except Exception as e:
            LOGGER.error(
                f"Invoke raster analysis lambda failed for aws request id: {context.aws_request_id}, tile: {tile}"
            )
            raise e
