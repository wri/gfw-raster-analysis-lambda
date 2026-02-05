import os

from pandas import DataFrame

from raster_analysis.data_cube import DataCube
from raster_analysis.data_environment import DataEnvironment
from raster_analysis.geometry import GeometryTile
from raster_analysis.globals import LOGGER
from raster_analysis.query import Query
from raster_analysis.query_executor import QueryExecutor
from raster_analysis.results_store import AnalysisResultsStore, ResultStatus

try:
    if os.getenv("DISABLE_CODEGURU", "").lower() in ("1", "true", "yes"):
        raise ImportError()
    from codeguru_profiler_agent import with_lambda_profiler
except ImportError:
    def with_lambda_profiler(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


@with_lambda_profiler(profiling_group_name="raster_analysis_default_profiler")
def handler(event, context):
    results_store = event.get("__bench_results_store__") or AnalysisResultsStore()

    cache_id = event.get("cache_id")
    if not cache_id:
        raise KeyError("Missing required field 'cache_id'")

    try:
        results = compute(event, context)
        LOGGER.debug("Ran analysis; cache_id=%s rows=%s cols=%s", cache_id, len(results), list(results.columns))

        results_store.save_result(results, cache_id)

    except Exception as e:
        LOGGER.exception(e)
        results_store.save_status(cache_id, ResultStatus.error, 0, str(e))

        raise


def compute(event, context) -> DataFrame:
    if "geometry" in event:
        source_geom = event["geometry"]
        is_encoded = False
    elif "encoded_geometry" in event:
        source_geom = event["encoded_geometry"]
        is_encoded = True
    else:
        raise KeyError("No valid geometry field")

    tile_geojson = event.get("tile", None)
    geom_tile = GeometryTile(source_geom, tile_geojson, is_encoded)

    if not geom_tile.geom:
        LOGGER.info("Geometry for tile %s is empty.", getattr(context, "aws_request_id", "unknown"))
        return DataFrame()

    data_environment = DataEnvironment(layers=event["environment"])
    query = Query(event["query"], data_environment)

    data_cube = DataCube(geom_tile.geom, geom_tile.tile, query)
    query_executor = QueryExecutor(query, data_cube)

    results: DataFrame = query_executor.execute()
    return results
