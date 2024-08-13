from aws_xray_sdk.core import patch, xray_recorder
from shapely.geometry import mapping
from shapely.wkb import loads

from raster_analysis.data_environment import DataEnvironment
from raster_analysis.exceptions import QueryParseException
from raster_analysis.geometry import decode_geometry
from raster_analysis.globals import LOGGER
from raster_analysis.tiling import AnalysisTiler

patch(["boto3"])


@xray_recorder.capture("Tiled Analysis")
def handler(event, context):
    try:
        LOGGER.info(f"Running analysis with parameters: {event}")

        query = event["query"]
        fid = event.get("fid", None)
        geojson = mapping(decode_geometry(event["geometry"]))
        data_environment = DataEnvironment(layers=event["environment"])

        LOGGER.info(f"Executing query: {query}")
        LOGGER.info(f"On geometry: {geojson}")

        tiler = AnalysisTiler(query, geojson, context.aws_request_id, data_environment)
        tiler.execute()

        results = tiler.result_as_dict()

        LOGGER.info("Successfully merged tiled results: {results}")
        response = {"status": "success", "data": results}
        if fid:
            response["fid"] = fid

        return response
    except QueryParseException as e:
        response = {"status": "failed", "message": str(e)}
        if fid:
            response["fid"] = fid

        return response
    except Exception as e:
        LOGGER.exception(e)
        response = {"status": "error", "message": str(e)}
        if fid:
            response["fid"] = fid

        return response
