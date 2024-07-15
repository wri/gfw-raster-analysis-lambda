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
        manifest = json.loads(response["Body"].read().decode("utf-8"))
        LOGGER.info("manifest file", manifest)

        combined_data = {}
        for result_record in manifest["ResultFiles"]["SUCCEEDED"]:
            response = s3_client().get_object(
                Bucket=results_meta["Bucket"], Key=result_record["Key"]
            )
            results = json.loads(response["Body"].read().decode("utf-8"))

            combined_data = {}
            for geom_result in results:
                result = geom_result["Output"]
                if result["status"] == "success":
                    fid = result["fid"]
                    combined_data[fid] = result["data"]

        LOGGER.info("Successfully aggregated results")
        return {"status": "success", "data": combined_data}
    except QueryParseException as e:
        return {"status": "failed", "message": str(e)}
    except Exception as e:
        LOGGER.exception(e)
        return {"status": "error", "message": str(e)}
