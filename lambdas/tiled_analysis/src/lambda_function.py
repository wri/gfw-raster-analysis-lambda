import os

from raster_analysis.data_environment import DataEnvironment
from raster_analysis.exceptions import QueryParseException
from raster_analysis.globals import LOGGER
from raster_analysis.tiling import AnalysisTiler

try:
    if os.getenv("DISABLE_CODEGURU", "").lower() in ("1", "true", "yes"):
        raise ImportError()
    from codeguru_profiler_agent import with_lambda_profiler
except ImportError:
    def with_lambda_profiler(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


@with_lambda_profiler(profiling_group_name="raster_analysis_tiled_profiler")
def handler(event, context):
    try:
        LOGGER.debug(f"Running analysis with parameters: {event}")

        query = event["query"]
        geojson = event["geometry"]
        format = event.get("format", "json")
        data_environment = DataEnvironment(layers=event["environment"])

        LOGGER.info(f"Executing query: {query}")
        LOGGER.debug(f"On geometry: {geojson}")

        tiler = AnalysisTiler(query, geojson, context.aws_request_id, data_environment)
        tiler.execute()

        LOGGER.info(f"Successfully executed query: {query}")

        if format == "csv":
            results = tiler.result_as_csv()
        else:
            results = tiler.result_as_dict()

        LOGGER.debug("Successfully merged tiled results")
        return {"status": "success", "data": results}
    except QueryParseException as e:
        return {"status": "failed", "message": str(e)}
    except Exception as e:
        LOGGER.exception(e)
        return {"status": "error", "message": str(e)}
