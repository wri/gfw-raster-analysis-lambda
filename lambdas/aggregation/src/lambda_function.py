from aws_xray_sdk.core import patch, xray_recorder

from raster_analysis.exceptions import QueryParseException
from raster_analysis.globals import LOGGER

patch(["boto3"])


@xray_recorder.capture("Aggregation")
def handler(event, context):
    try:
        LOGGER.info(f"Running aggregate with parameters: {event}")

        LOGGER.info("Successfully aggregated results")
        return {"status": "success"}
    except QueryParseException as e:
        return {"status": "failed", "message": str(e)}
    except Exception as e:
        LOGGER.exception(e)
        return {"status": "error", "message": str(e)}
