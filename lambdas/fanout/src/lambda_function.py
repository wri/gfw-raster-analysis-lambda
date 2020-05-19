import logging
from lambda_decorators import json_http_resp

from aws_xray_sdk.core import patch
from aws_xray_sdk.core import xray_recorder

from copy import deepcopy
import boto3
import os
import json

patch(["boto3"])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@json_http_resp
@xray_recorder.capture("Tiled Analysis")
def handler(event, context):
    tiles = event.get("tiles", [])
    payload_base = event["payload"]

    lambda_client = boto3.Session().client("lambda")
    for tile in tiles:
        payload = deepcopy(payload_base)
        payload["tile"] = tile

        lambda_response = lambda_client.invoke(
            FunctionName=os.environ["RASTER_ANALYSIS_LAMBDA_NAME"],
            InvocationType="Event",
            Payload=bytes(json.dumps(payload), "utf-8"),
        )

        if lambda_response["statusCode"] != 202:
            logger.error(
                f"Lambda invoke returned status {lambda_response['statusCode']}, aws request id: {context.aws_request_id}, geom: {geom}"
            )
            raise Exception(
                "Lambda invoke returned status {lambda_response['statusCode']}"
            )
