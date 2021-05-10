from raster_analysis.data_environment import DataEnvironment
from raster_analysis.exceptions import QueryParseException
from raster_analysis.globals import LOGGER
from raster_analysis.tiling import AnalysisTiler


def handler(event, context):
    try:
        query = event["query"]
        geojson = event["geometry"]
        format = event.get("format", "json")
        data_environment = DataEnvironment(layers=event["environment"])

        LOGGER.info(f"Executing query: {query}")

        tiler = AnalysisTiler(query, geojson, context.aws_request_id, data_environment)
        tiler.execute()

        if format == "csv":
            results = tiler.result_as_csv()
        else:
            results = tiler.result_as_dict()

        LOGGER.info("Successfully merged tiled results")
        return {"status": "success", "data": results}
    except QueryParseException as e:
        return {"status": "fail", "message": str(e)}
    except Exception as e:
        LOGGER.exception(e)
        return {"status": "error", "message": str(e)}
