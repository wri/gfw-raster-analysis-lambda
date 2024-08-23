from codeguru_profiler_agent import with_lambda_profiler
from pandas import DataFrame

from raster_analysis.data_cube import DataCube
from raster_analysis.data_environment import DataEnvironment
from raster_analysis.geometry import GeometryTile
from raster_analysis.globals import LOGGER
from raster_analysis.query import Query
from raster_analysis.query_executor import QueryExecutor
from raster_analysis.results_store import AnalysisResultsStore, ResultStatus


@with_lambda_profiler(profiling_group_name="raster_analysis_default_profiler")
def handler(event, context):
    try:
        LOGGER.info(f"Running analysis with parameters: {event}")
        results_store = AnalysisResultsStore()

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
            LOGGER.info(f"Geometry for tile {context.aws_request_id} is empty.")
            results_store.save_result(DataFrame(), context.aws_request_id)
            return {}

        data_environment = DataEnvironment(layers=event["environment"])
        query = Query(event["query"], data_environment)

        data_cube = DataCube(geom_tile.geom, geom_tile.tile, query)

        query_executor = QueryExecutor(query, data_cube)
        results: DataFrame = query_executor.execute()

        LOGGER.debug(f"Ran analysis with results: {results.head(100)}")
        results_store.save_result(results, event["cache_id"])
    except Exception as e:
        LOGGER.exception(e)

        results_store = AnalysisResultsStore()
        results_store.save_status(event["cache_id"], ResultStatus.error, 0, str(e))
        raise e
