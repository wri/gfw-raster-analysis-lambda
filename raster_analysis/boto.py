import boto3
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

S3_CLIENT = None
LAMBDA_CLIENT = None
DYN_CLIENT = None
DYN_RESOURCE = None
LAMBDA_CLIENT = None


def s3_client(proxy_endpoint=None):
    global S3_CLIENT
    if not S3_CLIENT:
        S3_CLIENT = boto3.client("s3", endpoint_url=proxy_endpoint)

    return S3_CLIENT


def dynamodb_client(proxy_endpoint=None):
    global DYN_CLIENT
    if not DYN_CLIENT:
        DYN_CLIENT = boto3.client("dynamodb", endpoint_url=proxy_endpoint)

    return DYN_CLIENT


def dynamodb_resource(proxy_endpoint=None):
    global DYN_RESOURCE
    if not DYN_RESOURCE:
        DYN_RESOURCE = boto3.resource("dynamodb", endpoint_url=proxy_endpoint)

    return DYN_RESOURCE


def lambda_client(proxy_endpoint=None):
    global S3_CLIENT
    if not S3_CLIENT:
        S3_CLIENT = boto3.client("lambda", endpoint_url=proxy_endpoint)

    return S3_CLIENT


def invoke_lambda(payload, lambda_name, client):
    response = client.invoke(
        FunctionName=lambda_name,
        InvocationType="Event",
        Payload=bytes(json.dumps(payload), "utf-8"),
    )

    if response["StatusCode"] != 202:
        logger.error(f"Status code: {response['status_code']}")
        raise AssertionError(f"Status code: {response['status_code']}")
