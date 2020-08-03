import boto3

S3_CLIENT = None
DYN_CLIENT = None
DYN_RESOURCE = None


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
