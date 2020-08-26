from copy import deepcopy

from aws_xray_sdk.core import patch
from aws_xray_sdk.core import xray_recorder

from raster_analysis.boto import lambda_client, invoke_lambda
from raster_analysis.globals import LOGGER, RASTER_ANALYSIS_LAMBDA_NAME

patch(["boto3"])


@xray_recorder.capture("Fanout")
def handler(event, context):
    tiles = event.get("tiles", [])
    payload_base = event["payload"]

    for tile in tiles:
        payload = deepcopy(payload_base)
        payload["tile"] = tile

        try:
            invoke_lambda(payload, RASTER_ANALYSIS_LAMBDA_NAME, lambda_client())
        except Exception as e:
            LOGGER.error(
                f"Invoke raster analysis lambda failed for aws request id: {context.aws_request_id}, tile: {tile}"
            )
            raise e
