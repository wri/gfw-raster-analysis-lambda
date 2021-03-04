from io import StringIO
from aws_xray_sdk.core import xray_recorder

from raster_analysis.results_store import AnalysisResultsStore
from raster_analysis.globals import LOGGER
from raster_analysis.data_cube import DataCube
from raster_analysis.query_executor import QueryExecutor
from raster_analysis.geometry import GeometryTile
from raster_analysis.query import Query, QueryParseException


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
        query_executor.execute()
        csv_results: StringIO = query_executor.result_as_csv()

        csv_str = csv_results.getvalue()
        LOGGER.info(f"Ran analysis with results: {csv_str}")
        results_store.save_result(csv_str, context.aws_request_id)
    except Exception as e:
        results_store = AnalysisResultsStore(event["analysis_id"])
        results_store.save_error(context.aws_request_id)

        LOGGER.exception(e)
        raise Exception(f"Internal Server Error <{context.aws_request_id}>")

