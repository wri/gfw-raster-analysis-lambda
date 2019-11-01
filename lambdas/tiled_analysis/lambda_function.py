from shapely.geometry import shape, mapping, box

import pandas as pd
import boto3
import json

import threading
import queue

import logging

from aws_xray_sdk.core import patch

patch(["boto3"])

TILE_WIDTH = 1.25

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    geom = shape(event["geometry"])
    tiles = get_tiles(geom, TILE_WIDTH)
    intersecting_geoms = []

    for tile in tiles:
        inter_geom = tile.intersection(geom)

        if not inter_geom.is_empty:
            intersecting_geoms.append(tile.intersection(geom))

    execution_threads = []
    result_queue = queue.Queue()
    error_queue = queue.Queue()

    for geom in intersecting_geoms:
        payload = event.copy()
        payload["geometry"] = mapping(geom)

        execution_thread = threading.Thread(
            target=execute_raster_analysis_lambda,
            args=(payload, result_queue, error_queue),
        )
        execution_thread.start()
        execution_threads.append(execution_thread)

    for execution_thread in execution_threads:
        execution_thread.join()

    errors = [error_queue.get() for _ in range(error_queue.qsize())]
    if errors:
        for e in errors:
            logger.error(e)
        return {
            "statusCode": 500,
            "body": "Internal Server Error <" + context.aws_request_id + ">",
        }

    results = [result_queue.get() for _ in range(result_queue.qsize())]
    change_tables = [result[0] for result in results]
    nonempty_change_tables = filter(
        lambda change_table: not change_table.empty, change_tables
    )
    summary_tables = [result[1] for result in results]

    result = dict()
    result["change_table"] = merge_change_tables(
        nonempty_change_tables, event
    ).to_dict()

    if "get_area_summary" in event and event["get_area_summary"] is True:
        result["summary_table"] = merge_summary_tables(summary_tables)

    return {"statusCode": 200, "body": json.dumps(result)}


def merge_change_tables(change_tables, event):
    groupby_columns = [event["analysis_raster_id"]]
    if "contextual_raster_ids" in event:
        groupby_columns += event["contextual_raster_ids"]

    return pd.concat(change_tables).groupby(groupby_columns).sum().reset_index()


def merge_summary_tables(summary_tables):
    merged_summary_table = dict()

    for summary_table in summary_tables:
        for col, result in summary_table.items():
            if col in merged_summary_table:
                merged_summary_table[col] += result
            else:
                merged_summary_table[col] = result

    return merged_summary_table


def execute_raster_analysis_lambda(payload, result_queue, error_queue):
    try:
        lambda_client = boto3.Session().client("lambda")

        lambda_response = lambda_client.invoke(
            FunctionName="raster-analysis",
            InvocationType="RequestResponse",
            Payload=bytes(json.dumps(payload), "utf-8"),
        )

        response = json.loads(lambda_response["Payload"].read())
        result = json.loads(response["body"])

        if response["statusCode"] == 200:
            change_table = pd.DataFrame.from_dict(result["change_table"])
            summary_table = (
                result["summary_table"] if "summary_table" in result else None
            )
            result_queue.put((change_table, summary_table))
        else:
            error_queue.put((response["statusCode"], result))
    except Exception as e:
        logging.error(e)
        error_queue.put(e)


def get_tiles(geom, width):
    min_x, min_y, max_x, max_y = get_rounded_bounding_box(geom, width)
    tiles = []

    for i in range(0, int((max_x - min_x) / width)):
        for j in range(0, int((max_y - min_y) / width)):
            tiles.append(
                box(
                    (i * width) + min_x,
                    (j * width) + min_y,
                    ((i + 1) * width) + min_x,
                    ((j + 1) * width) + min_y,
                )
            )

    return tiles


def get_rounded_bounding_box(geom, width):
    return (
        geom.bounds[0] - (geom.bounds[0] % width),
        geom.bounds[1] - (geom.bounds[1] % width),
        geom.bounds[2] + (-geom.bounds[2] % width),
        geom.bounds[3] + (-geom.bounds[3] % width),
    )


if __name__ == "__main__":
    print(
        lambda_handler(
            {
                "analysis_raster_id": "loss",
                "contextual_raster_ids": ["wdpa"],
                "analyses": ["count", "area"],
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [20.4374999999624, 2.30604814085596],
                            [19.2949218749763, 0.1972956024577],
                            [20.9648437499629, -1.73603206689361],
                            [25.0078124999669, -1.38460047301014],
                            [24.8320312499668, 1.95472918639642],
                            [20.4374999999624, 2.30604814085596],
                        ]
                    ],
                },
                "get_area_summary": True,
                "filter_raster_id": "tcd_2000",
                "filter_intervals": [[0, 30]],
            },
            None,
        )
    )
