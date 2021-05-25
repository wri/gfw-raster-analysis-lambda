from aws_xray_sdk.core import patch
from aws_xray_sdk.core import xray_recorder

from raster_analysis.exceptions import QueryParseException
from raster_analysis.tiling import AnalysisTiler
from raster_analysis.globals import LOGGER

patch(["boto3"])


@xray_recorder.capture("Tiled Analysis")
def handler(event, context):
    try:
        query = event["query"]
        geojson = event["geometry"]
        format = event.get("format", "json")

        LOGGER.info(f"Executing query: {query}")

        tiler = AnalysisTiler(query, geojson, context.aws_request_id)
        tiler.execute()

        if format == "csv":
            results = tiler.result_as_csv()
        else:
            results = tiler.result_as_dict()

        LOGGER.info("Successfully merged tiled results")
        return {"status": "success", "data": results}
    except QueryParseException as e:
        return {"status": "failed", "message": str(e)}
    except Exception as e:
        LOGGER.exception(e)
        return {"status": "error", "message": str(e)}
