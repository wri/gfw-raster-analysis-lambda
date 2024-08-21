import json

from aws_xray_sdk.core import patch, xray_recorder

from raster_analysis.boto import s3_client
from raster_analysis.globals import LOGGER

patch(["boto3"])


@xray_recorder.capture("Aggregation")
def handler(event, context):
    id_field = event["id_field"]
    query = event["query"]
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
                        {"result": result["data"], id_field: result["fid"]}
                    )
                else:
                    failed_geometries.append(
                        {id_field: result["fid"], "detail": result["message"]}
                    )

        for failed_record in manifest["ResultFiles"]["FAILED"]:
            response = s3_client().get_object(Bucket=bucket, Key=failed_record["Key"])
            errors = json.loads(response["Body"].read().decode("utf-8"))
            for error in errors:
                input = json.loads(error["Input"])
                if error["Status"] == "FAILED":
                    failed_geometries.append(
                        {id_field: input["fid"], "detail": error["Error"]}
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

        s3_client().put_object(
            Bucket=bucket,
            Key=failed_list_key,
            Body=json.dumps(failed_geometries),
        )

        # get status and links based on results
        if combined_data:
            download_link = f"s3://{bucket}/{results_key}"

            # partial success if there are both success and failures present
            if failed_geometries:
                status = "partial_success"
                failed_geometries_link = f"s3://{bucket}/{failed_list_key}"
            else:
                status = "success"
                failed_geometries_link = None
        else:
            download_link = None

            if failed_geometries:
                status = "failed"
                failed_geometries_link = f"s3://{bucket}/{failed_list_key}"
            else:
                # error if there are both are empty, something strange happened
                status = "error"
                failed_geometries_link = None

        return {
            "status": status,
            "query": query,
            "data": {
                "download_link": download_link,
                "failed_geometries_link": failed_geometries_link,
            },
        }
    except Exception as e:
        LOGGER.exception(e)
        return {"status": "error", "message": str(e)}
