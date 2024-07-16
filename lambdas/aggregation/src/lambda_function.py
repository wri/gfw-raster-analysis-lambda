import json

from datetime import datetime

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

        combined_data = []
        failed_geometries = []
        for result_record in manifest["ResultFiles"]["SUCCEEDED"]:
            response = s3_client().get_object(
                Bucket=results_meta["Bucket"], Key=result_record["Key"]
            )
            results = json.loads(response["Body"].read().decode("utf-8"))
            for geom_result in results:
                result = json.loads(geom_result["Output"])
                if result["status"] == "success":
                    combined_data.append(
                        {"result": result["data"], "geometry_id": result["fid"]}
                    )
                else:
                    failed_geometries.append(
                        {"geometry_id": result["fid"], "detail": result["message"]}
                    )

        LOGGER.info("Successfully aggregated results")

        results_prefix = "/".join(results_meta["Key"].split("/")[:-1])
        results_key = f"{results_prefix}/analysis_results.json"
        failed_list_key = f"{results_prefix}/failed_geometries.json"

        s3_client().put_object(
            Bucket=results_meta["Bucket"],
            Key=results_key,
            Body=json.dumps(combined_data),
        )
        s3_client().put_object(
            Bucket=results_meta["Bucket"],
            Key=failed_list_key,
            Body=json.dumps(failed_geometries),
        )

        return {
            "status": "success",
            "data": {
                "analysis_results": f"s3://{results_meta['Bucket']}/{results_key}",
                "failed_geometries": f"s3://{results_meta['Bucket']}/{failed_list_key}",
            },
        }
    except QueryParseException as e:
        return {"status": "failed", "message": str(e)}
    except Exception as e:
        LOGGER.exception(e)
        return {"status": "error", "message": str(e)}
