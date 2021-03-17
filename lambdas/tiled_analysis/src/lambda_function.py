from aws_xray_sdk.core import patch
from aws_xray_sdk.core import xray_recorder

from raster_analysis.tiling import AnalysisTiler
from raster_analysis.globals import LOGGER

patch(["boto3"])


@xray_recorder.capture("Tiled Analysis")
def handler(event, context):
    try:
        query = event["query"]
        geojson = event["geometry"]
        format = event.get("format", "json")

        tiler = AnalysisTiler(query, geojson, context.aws_request_id)
        tiler.execute()

        if format == "csv":
            results = tiler.result_as_csv()
        else:
            results = tiler.result_as_dict()

        LOGGER.info("Successfully merged tiled results")
        return {"statusCode": 200, "body": {"status": "success", "data": results}}
    except Exception as e:
        LOGGER.exception(e)
        return {"statusCode": 500, "body": {"status": "failed", "message": str(e)}}
