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

    contextual_raster_ids = (
        event["contextual_raster_ids"] if "contextual_raster_ids" in event else []
    )

    change_groupby_columns = [event["analysis_raster_id"]] + contextual_raster_ids
    result["detailed_table"] = merge_tables(
        nonempty_change_tables, change_groupby_columns
    ).to_dict()

    if "area" in event["analyses"]:
        nonempty_summary_tables = filter(
            lambda summary_table: not summary_table.empty, summary_tables
        )
        result["summary_table"] = merge_summary_tables(
            nonempty_summary_tables, contextual_raster_ids
        ).to_dict()

    return {"statusCode": 200, "body": json.dumps(result)}


def merge_tables(tables, groupby_columns):
    return pd.concat(tables).groupby(groupby_columns).sum().reset_index()


def merge_summary_tables(tables, contextual_raster_ids):
    if contextual_raster_ids:
        return merge_tables(tables, contextual_raster_ids)
    else:
        return pd.DataFrame(pd.concat(tables).sum()).transpose()


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
            change_table = pd.DataFrame.from_dict(result["detailed_table"])
            summary_table = (
                pd.DataFrame.from_dict(result["summary_table"])
                if "summary_table" in result
                else None
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
                "contextual_raster_ids": ["wdpa", "ifl"],
                "analyses": ["count", "area"],
                "aggregate_raster_ids": ["biomass"],
                "density_raster_ids": ["biomass"],
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-56.9291420118343, -10.461390532544382],
                            [-54.29482248520708, -10.287426035502962],
                            [-52.39363905325442, -10.759615384615389],
                            [-52.045710059171576, -13.55547337278107],
                            [-54.26997041420116, -15.630621301775152],
                            [-57.00369822485205, -14.673816568047341],
                            [-57.84866863905324, -12.350147928994087],
                            [-56.9291420118343, -10.461390532544382],
                        ]
                    ],
                },
                "filter_raster_id": "tcd_2000",
                "filter_intervals": [[0, 30]],
            },
            None,
        )
    )
