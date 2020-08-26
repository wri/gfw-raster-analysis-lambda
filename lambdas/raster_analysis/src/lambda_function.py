from typing import Optional
from datetime import datetime

from shapely.geometry import shape
from aws_xray_sdk.core import xray_recorder

from raster_analysis.results_store import AnalysisResultsStore
from raster_analysis.globals import LOGGER
from raster_analysis.layer.data_cube import DataCube


@xray_recorder.capture("Raster Analysis Lambda")
def handler(event, context):
    try:
        LOGGER.info(f"Running analysis with parameters: {event}")
        geometry = shape(event["geometry"])

        if "tile" in event:
            tile = shape(event["tile"])
            geometry = geometry.intersection(tile)

        start_date = try_parsing_date(event.get("start_date", None))
        end_date = try_parsing_date(event.get("end_date", None))

        data_cube = DataCube(
            geometry,
            tile,
            event.get("group_by", []),
            event.get("sum", []),
            event.get("filters", []),
            start_date,
            end_date,
        )
        result = data_cube.calculate()
        LOGGER.info(f"Ran analysis with result: {result}")

        results_store = AnalysisResultsStore(event["analysis_id"])
        results_store.save_result(result, context.aws_request_id)

        return result
    except Exception as e:
        LOGGER.exception(e)
        results_store.save_result()
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
