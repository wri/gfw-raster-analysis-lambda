from shapely.geometry import shape
import shapely.wkt

from lambda_decorators import json_http_resp

from aws_xray_sdk.core import patch
from aws_xray_sdk.core import xray_recorder

import logging
from datetime import datetime, date, timedelta

from raster_analysis.tiling import (
    get_tiles,
    get_intersecting_geoms,
    process_tiled_geoms,
    merge_tile_results,
)
from raster_analysis.exceptions import RasterAnalysisException

patch(["boto3"])

TILE_WIDTH = 1
GLAD_UNCONFIRMED_CONST = 20000
GLAD_CONFIRMED_CONST = 30000

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# @json_http_resp
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

    tile_results = process_tiled_geoms(tiles, event, context.aws_request_id, 20)

    # try:
    #    tile_results = process_tiled_geoms(tiles, event, context.aws_request_id, 20)
    # except RasterAnalysisException:
    #    return {
    #        "statusCode": 500,
    #        "body": "Internal Server Error <" + context.aws_request_id + ">",
    #    }

    logger.info(f"Successfully ran tiled analysis with results: {tile_results}")

    result = merge_tile_results(tile_results, group_by)

    logger.info(f"Successfully merged tiled results to produce final result: {result}")

    return convert_to_csv_json_style(result)


def convert_to_csv_json_style(results):
    result_cols = list(results.keys())
    result_col_sample = results[result_cols[0]]

    if not isinstance(result_col_sample, list):
        return results

    result_row_length = len(results[result_cols[0]])
    rows = []

    for i in range(0, result_row_length):
        row = dict()
        for col in result_cols:
            if col == "glad_alerts":
                days_since_2015 = results[col][i] - GLAD_CONFIRMED_CONST
                raw_date = date(2015, 1, 1) + timedelta(days=days_since_2015)
                row["alert__date"] = raw_date.strftime("%Y-%m-%d")
            else:
                row[col] = results[col][i]

        rows.append(row)

    return rows
