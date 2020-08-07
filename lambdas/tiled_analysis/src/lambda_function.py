from shapely.geometry import shape
import shapely.wkt

from aws_xray_sdk.core import patch
from aws_xray_sdk.core import xray_recorder

import logging

from raster_analysis.exceptions import RasterAnalysisException
from raster_analysis.tiling import get_tiles, process_tiled_geoms, merge_tile_results

patch(["boto3"])

TILE_WIDTH = 1

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@xray_recorder.capture("Tiled Analysis")
def handler(event, context):
    geom = shape(event["geometry"])
    group_by = event.get("group_by", [])

    logger.info(
        f"Tiling input geometry with width={TILE_WIDTH}: {shapely.wkt.dumps(geom)}"
    )
    tiles = get_tiles(geom, TILE_WIDTH)

    logger.info(
        f"Running tiled analysis on the following tiles: {[shapely.wkt.dumps(g) for g in tiles]}"
    )

    try:
        tile_results = process_tiled_geoms(tiles, event, context.aws_request_id, 20)
    except RasterAnalysisException:
        return {
            "statusCode": 500,
            "body": "Internal Server Error <" + context.aws_request_id + ">",
        }

    logger.info(f"Successfully ran tiled analysis with results: {tile_results}")

    result = merge_tile_results(tile_results, group_by)

    logger.info(f"Successfully merged tiled results to produce final result: {result}")

    return result
