from typing import Optional
from datetime import datetime

from shapely.geometry import shape
from aws_xray_sdk.core import xray_recorder

from raster_analysis.exceptions import InvalidGeometryException
from raster_analysis.results_store import AnalysisResultsStore
from raster_analysis.globals import LOGGER
from raster_analysis.data_cube import DataCube
from raster_analysis.utils import decode_geometry
from raster_analysis.query_executor import QueryResult, QueryExecutor


@xray_recorder.capture("Raster Analysis Lambda")
def handler(event, context):
    try:
        LOGGER.info(f"Running analysis with parameters: {event}")
        results_store = AnalysisResultsStore(event["analysis_id"])

        geojson = event.get("geometry", None)
        if geojson:
            geometry = shape(geojson)
        else:
            geometry = decode_geometry(event["encoded_geometry"])

        if "tile" in event:
            tile = shape(event["tile"])
            geometry = geometry.intersection(tile)

            if not geometry.is_valid:
                geometry = geometry.buffer(0)

                if not geometry.is_valid:
                    raise InvalidGeometryException(
                        f"Geometry {geometry.wkt} is invalid"
                    )

            if geometry.is_empty:
                LOGGER.info(f"Geometry for tile {context.aws_request_id} is empty.")
                results_store.save_result({}, context.aws_request_id)
                return {}

        data_cube = DataCube(geometry, tile, query)
        query_executor = QueryExecutor(query, data_cube)
        results: QueryResult = query_executor.execute()

        LOGGER.info(f"Ran analysis with results: {results}")
        results_store.save_result(results, context.aws_request_id)
    except Exception as e:
        results_store = AnalysisResultsStore(event["analysis_id"])
        results_store.save_error(context.aws_request_id)

        LOGGER.exception(e)
        raise Exception(f"Internal Server Error <{context.aws_request_id}>")


def try_parsing_date(text: str) -> Optional[datetime]:
    if text:
        for fmt in ("%Y-%m-%d", "%Y"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass
        raise ValueError("no valid date format found")
    return None
