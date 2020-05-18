from shapely.geometry import shape
import shapely.wkt

import logging
from lambda_decorators import json_http_resp

from aws_xray_sdk.core import patch
from aws_xray_sdk.core import xray_recorder

from raster_analysis.tiling import (
    get_tiles,
    get_intersecting_geoms,
    process_tiled_geoms_async,
    merge_tile_results,
)
from raster_analysis.exceptions import RasterAnalysisException

patch(["boto3"])

TILE_WIDTH = 1.25

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@json_http_resp
@xray_recorder.capture("Tiled Analysis")
def handler(event, context):
    geom = shape(event["geometry"])
    contextual_raster_ids = (
        event["contextual_raster_ids"] if "contextual_raster_ids" in event else []
    )
    analysis_raster_id = (
        [event["analysis_raster_id"]] if "analysis_raster_id" in event else []
    )

    logger.info(
        f"Tiling input geometry with width={TILE_WIDTH}: {shapely.wkt.dumps(geom)}"
    )
    tiles = get_tiles(geom, TILE_WIDTH)
    tiled_geoms = get_intersecting_geoms(geom, tiles)

    logger.info(
        f"Running tiled analysis on the following geometries: {[shapely.wkt.dumps(g) for g in tiled_geoms]}"
    )

    try:
        tile_results = process_tiled_geoms_async(
            tiled_geoms, event, context.aws_request_id, 20
        )
    except RasterAnalysisException:
        return {
            "statusCode": 500,
            "body": "Internal Server Error <" + context.aws_request_id + ">",
        }

    logger.info(f"Successfully ran tiled analysis with results: {tile_results}")

    groupby_columns = analysis_raster_id + contextual_raster_ids
    result = merge_tile_results(tile_results, groupby_columns)

    logger.info(
        f"Successfully merged tiled results to produce final result: {tile_results}"
    )

    return result


if __name__ == "__main__":
    """
    import random

    class Context(object):
        pass

    context = Context()
    context.aws_request_id = f"test_id_{random.randint(0, 1000)}"
    context.log_stream_name = "test_log_stream"
    """

    print(
        handler(
            {
                "analyses": ["count", "area"],
                "analysis_raster_id": "loss",
                "contextual_raster_ids": ["wdpa"],
                # "aggregate_raster_ids": ["biomass"],
                "extent_year": 2000,
                "threshold": 30,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-64.6929931640625, -11.770570195625227],
                            [-62.4078369140625, -11.770570195625227],
                            [-62.4078369140625, -10.260870794748941],
                            [-64.6929931640625, -10.260870794748941],
                            [-64.6929931640625, -11.770570195625227],
                        ]
                    ],
                },
            },
            None,
        )
    )
