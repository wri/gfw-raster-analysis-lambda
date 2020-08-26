from shapely.geometry import shape
from aws_xray_sdk.core import patch
from aws_xray_sdk.core import xray_recorder
import shapely.wkt

from raster_analysis.tiling import get_tiles, process_tiled_geoms, merge_tile_results
from raster_analysis.globals import LOGGER, TILE_WIDTH

patch(["boto3"])


@xray_recorder.capture("Tiled Analysis")
def handler(event, context):
    try:
        geom = shape(event["geometry"])
        group_by = event.get("group_by", [])

        LOGGER.debug(
            f"Tiling input geometry with width={TILE_WIDTH}: {shapely.wkt.dumps(geom)}"
        )
        tiles = get_tiles(geom, TILE_WIDTH)

        LOGGER.debug(
            f"Running tiled analysis on the following tiles: {[shapely.wkt.dumps(g) for g in tiles]}"
        )

        tile_results = process_tiled_geoms(tiles, event, context.aws_request_id, 20)
        LOGGER.info(f"Successfully ran tiled analysis with results: {tile_results}")

        result = merge_tile_results(tile_results, group_by)
        LOGGER.info(
            f"Successfully merged tiled results to produce final result: {result}"
        )

        return {"statusCode": 200, "body": {"status": "success", "data": result}}
    except Exception as e:
        LOGGER.exception(e)
        return {"statusCode": 500, "body": {"status": "failed", "message": str(e)}}
