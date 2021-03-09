from aws_xray_sdk.core import xray_recorder
from pandas import DataFrame

from raster_analysis.results_store import AnalysisResultsStore, ResultStatus
from raster_analysis.globals import LOGGER
from raster_analysis.data_cube import DataCube
from raster_analysis.query_executor import QueryExecutor
from raster_analysis.geometry import GeometryTile
from raster_analysis.query import Query


@xray_recorder.capture("Raster Analysis Lambda")
def handler(event, context):
    try:
        LOGGER.info(f"Running analysis with parameters: {event}")
        results_store = AnalysisResultsStore(event["analysis_id"])

        source_geom = event.get("geometry", None)
        tile_geojson = event.get("tile", None)
        is_encoded = event.get("is_encoded", False)

        geom_tile = GeometryTile(source_geom, tile_geojson, is_encoded)

        if not geom_tile.geom:
            LOGGER.info(f"Geometry for tile {context.aws_request_id} is empty.")
            results_store.save_result({}, context.aws_request_id)
            return {}

        query = Query.parse_query(event["query"])
        data_cube = DataCube(geom_tile.geom, geom_tile.tile, query)
        query_executor = QueryExecutor(query, data_cube)
        results: DataFrame = query_executor.execute()

        LOGGER.info(f"Ran analysis with results: {results}")
        results_store.save_result(results, context.aws_request_id)
    except Exception as e:
        results_store = AnalysisResultsStore(event["analysis_id"])
        results_store.save_status(context.aws_request_id, ResultStatus.error)

        LOGGER.exception(e)
        raise Exception(f"Internal Server Error <{context.aws_request_id}>")

