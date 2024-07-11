import json

from aws_xray_sdk.core import patch, xray_recorder

from raster_analysis.exceptions import QueryParseException
from raster_analysis.globals import LOGGER
from raster_analysis.boto import s3_client

patch(["boto3"])


@xray_recorder.capture("Aggregation")
def handler(event, context):
    results_meta = event["distributed_map"]["ResultWriterDetails"]
    try:
        LOGGER.info(f"Running aggregate with parameters: {event}")
        response = s3_client().get_object(
            Bucket=results_meta["Bucket"], Key=results_meta["Key"]
        )
        print("response", response)
        manifest = json.loads(response["Body"].read().decode("utf-8"))
        LOGGER.info("manifest file", manifest)

        for result_record in manifest["ResultFiles"]["SUCCEEDED"]:
            response = s3_client().get_object(
                Bucket=results_meta["Bucket"], Key=result_record["Key"]
            )
            results = json.loads(response["Body"].read().decode("utf-8"))

            LOGGER.info("results", results)

        LOGGER.info("Successfully aggregated results")
        return {"status": "success"}
    except QueryParseException as e:
        return {"status": "failed", "message": str(e)}
    except Exception as e:
        LOGGER.exception(e)
        return {"status": "error", "message": str(e)}
