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
        bucket = results_meta["Bucket"]
        LOGGER.info(f"Running aggregate with parameters: {event}")
        response = s3_client().get_object(Bucket=bucket, Key=results_meta["Key"])
        manifest = json.loads(response["Body"].read().decode("utf-8"))
        LOGGER.info("manifest file", manifest)

        combined_data = []
        failed_geometries = []
        for result_record in manifest["ResultFiles"]["SUCCEEDED"]:
            response = s3_client().get_object(Bucket=bucket, Key=result_record["Key"])
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
            Bucket=bucket,
            Key=results_key,
            Body=json.dumps(combined_data),
        )

        expires_in = 86400 * 5  # five days
        result_presigned_url = s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": results_key},
            ExpiresIn=expires_in,
        )
        s3_client().put_object(
            Bucket=bucket,
            Key=failed_list_key,
            Body=json.dumps(failed_geometries),
        )
        failed_list_presigned_url = s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": failed_list_key},
            ExpiresIn=expires_in,
        )

        return {
            "status": "success",
            "data": {
                "download_link": result_presigned_url,
                "failed_geometries_link": failed_list_presigned_url,
            },
        }
    except QueryParseException as e:
        return {"status": "failed", "message": str(e)}
    except Exception as e:
        LOGGER.exception(e)
        return {"status": "error", "message": str(e)}
