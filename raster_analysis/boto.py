import json
from typing import Any, Callable, Dict

import boto3

from raster_analysis.globals import (
    AWS_REGION,
    DYNAMODB_ENDPOINT_URL,
    LAMBDA_ENDPOINT_URL,
    LOGGER,
    S3_ENDPOINT_URL,
)


def client_constructor(
    service: str, entrypoint_url: str = None, type: str = "client"
) -> Callable:
    """Using closure design for a client constructor This way we only need to
    create the client once in central location and it will be easier to
    mock."""
    service_client = None

    def client():
        nonlocal service_client
        if service_client is None:
            if type == "resource":
                service_client = boto3.resource(
                    service, region_name=AWS_REGION, endpoint_url=entrypoint_url
                )
            else:
                service_client = boto3.client(
                    service, region_name=AWS_REGION, endpoint_url=entrypoint_url
                )

        return service_client

    return client


s3_client = client_constructor("s3", S3_ENDPOINT_URL)
lambda_client = client_constructor("lambda", LAMBDA_ENDPOINT_URL)
dynamodb_client = client_constructor("dynamodb", DYNAMODB_ENDPOINT_URL)
dynamodb_resource = client_constructor("dynamodb", DYNAMODB_ENDPOINT_URL, "resource")
sfn_client = client_constructor("stepfunctions", LAMBDA_ENDPOINT_URL)


def invoke_lambda(payload: Dict[str, Any], lambda_name: str, client) -> None:
    response = client.invoke(
        FunctionName=lambda_name,
        InvocationType="Event",
        Payload=bytes(json.dumps(payload), "utf-8"),
    )

    if response["StatusCode"] != 202:
        LOGGER.error(f"Status code: {response['status_code']}")
        raise AssertionError(f"Status code: {response['status_code']}")
