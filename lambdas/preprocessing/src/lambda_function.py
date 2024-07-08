from aws_xray_sdk.core import patch, xray_recorder

from raster_analysis.exceptions import QueryParseException
from raster_analysis.globals import LOGGER

patch(["boto3"])


@xray_recorder.capture("Preprocessing")
def handler(event, context):
    try:
        LOGGER.info(f"Running preprocessing with parameters: {event}")

        return {"status": "success",
                "geometries": {
                    "bucket": "s3://gfw-pipelines-test/test/otf_lists",
                    "key": "geometries.csv"
                },
                "output": {
                    "bucket": "s3://gfw-pipelines-test/test/otf_lists",
                    "prefix": "output"
                }
                }
    except QueryParseException as e:
        return {"status": "failed", "message": str(e)}
    except Exception as e:
        LOGGER.exception(e)
        return {"status": "error", "message": str(e)}
