from codeguru_profiler_agent import with_lambda_profiler

from raster_analysis.data_environment import DataEnvironment
from raster_analysis.exceptions import QueryParseException
from raster_analysis.globals import LOGGER
from raster_analysis.tiling import AnalysisTiler


@with_lambda_profiler(profiling_group_name="raster_analysis_tiled_profiler")
def handler(event, context):
    try:
        LOGGER.info(f"Running analysis with parameters: {event}")

        query = event["query"]
        geojson = event["geometry"]
        format = event.get("format", "json")
        data_environment = DataEnvironment(layers=event["environment"])

        LOGGER.info(f"Executing query: {query}")
        LOGGER.info(f"On geometry: {geojson}")

        tiler = AnalysisTiler(query, geojson, context.aws_request_id, data_environment)
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
