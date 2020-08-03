from shapely.geometry import mapping, box

import boto3
import json
import logging
import os
from datetime import date

from time import sleep

from copy import deepcopy

import numpy as np
from aws_xray_sdk.core import xray_recorder

from raster_analysis.boto import dynamodb_resource
import pandas

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@xray_recorder.capture("Merge Tiled Geometry Results")
def merge_tile_results(tile_results, groupby_columns):
    concatted_tile_results = concat_tile_results(tile_results)

    if not groupby_columns:
        return concatted_tile_results

    group_by_results = np.array(
        [concatted_tile_results[col] for col in groupby_columns]
    )
    column_maxes = [col.max() + 1 for col in group_by_results]
    linear_indices = np.ravel_multi_index(group_by_results, column_maxes).astype(
        np.uint32
    )

    unique_values, inv = np.unique(linear_indices, return_inverse=True)
    unique_value_combinations = np.unravel_index(unique_values, column_maxes)

    merged_tile_results = dict(zip(groupby_columns, unique_value_combinations))

    for col_name, col_data in concatted_tile_results.items():
        if col_name not in groupby_columns:
            arr = np.array(col_data)
            merged_tile_results[col_name] = np.bincount(inv, weights=col_data).astype(
                arr.dtype
            )

    # convert ordinal dates to readable dates
    for col in groupby_columns:
        if "__date" in col:
            merge_tile_results[col] = [
                date.fromordinal(val).strftime("%Y-%m-%d")
                for val in merge_tile_results[col]
            ]
        elif "__isoweek" in col:
            iso_dates = [
                date.fromordinal(val).isocalendar() for val in merge_tile_results[col]
            ]
            merge_tile_results[groupby_columns] = [d[1] for d in iso_dates]
            merge_tile_results[groupby_columns.replace("__isoweek", "__year")] = [
                d[0] for d in iso_dates
            ]

    for col, arr in merged_tile_results.items():
        merged_tile_results[col] = arr.tolist()

    return merged_tile_results


def concat_tile_results(tile_results):
    result = deepcopy(tile_results[0])

    for col in result.keys():
        for table in tile_results[1:]:
            result[col] += table[col]

    return result


def process_tiled_geoms(tiles, geoprocessing_params, request_id, fanout_num):
    geom_count = len(tiles)
    geoprocessing_params["write_to_dynamo"] = True
    geoprocessing_params["analysis_id"] = request_id
    logger.info(f"Processing {geom_count} tiles")

    tile_geojsons = [mapping(tile) for tile in tiles]
    tile_chunks = [
        tile_geojsons[x : x + fanout_num]
        for x in range(0, len(tile_geojsons), fanout_num)
    ]
    lambda_client = boto3.Session().client("lambda")
    fanout_lambda = os.environ["FANOUT_LAMBDA"]

    for chunk in tile_chunks:
        event = {"payload": geoprocessing_params, "tiles": chunk}
        invoke_lambda(event, fanout_lambda, lambda_client)

    curr_count = 0
    tries = 0

    tiled_table = dynamodb_resource().Table(os.environ["TILED_RESULTS_TABLE_NAME"])
    logger.info(f"Geom count: {geom_count}")
    while curr_count < geom_count and tries < 6000:
        sleep(0.5)
        tries += 1

        response = tiled_table.query(
            ExpressionAttributeValues={":id": request_id},
            KeyConditionExpression=f"analysis_id = :id",
            TableName=os.environ["TILED_RESULTS_TABLE_NAME"],
        )

        curr_count = response["Count"]

    if curr_count != geom_count:
        raise TimeoutError(
            f"Timeout occurred before all lambdas completed. Tile count: {geom_count}; tiles completed: {curr_count}"
        )

    results = [convert_from_decimal(item["result"]) for item in response["Items"]]
    return results


@xray_recorder.capture("Convert DynamoDB results")
def convert_from_decimal(raster_analysis_result):
    """
    DynamoDB API returns Decimal objects for all numbers, so this is a util to
    convert the result back to ints and floats

    :param result: resulting dict from a call to raster analysis
    :return: result with int and float values instead of Decimal
    """
    result = deepcopy(raster_analysis_result)

    for layer, col in result.items():
        if isinstance(col, list):
            if all([val % 1 == 0 for val in col]):
                result[layer] = [int(val) for val in col]
            else:
                result[layer] = [float(val) for val in col]
        else:
            result[layer] = int(col) if col % 1 == 0 else float(col)

    return result


def invoke_lambda(payload, lambda_name, lambda_client):
    response = lambda_client.invoke(
        FunctionName=lambda_name,
        InvocationType="Event",
        Payload=bytes(json.dumps(payload), "utf-8"),
    )

    if response["StatusCode"] != 202:
        logger.error(f"Status code: {response['status_code']}")
        raise AssertionError(f"Status code: {response['status_code']}")


@xray_recorder.capture("Get Tiles")
def get_tiles(geom, width):
    """
    Get width x width tile geometries over the extent of the geometry
    TODO if there's a multipolygon (e.g. from shapefile) where polygons are dotted across the world,
    TODO will this take forever to check intersections? Seems uncommon, but can probably just
    TODO look at each individual polygon (but then what if there's a multipolygon with a ridiculous
    TODO number of small polygons?
    """
    min_x, min_y, max_x, max_y = _get_rounded_bounding_box(geom, width)
    tiles = []

    for i in range(0, int((max_x - min_x) / width)):
        for j in range(0, int((max_y - min_y) / width)):
            tile = box(
                (i * width) + min_x,
                (j * width) + min_y,
                ((i + 1) * width) + min_x,
                ((j + 1) * width) + min_y,
            )

            if geom.intersects(tile):
                tiles.append(tile)

    return tiles


@xray_recorder.capture("Get Intersecting Geometries")
def get_intersecting_geoms(geom, tiles):
    """
    Divide geom into geoms intersected with the tiles
    """
    intersecting_geoms = []
    for tile in tiles:
        inter_geom = tile.intersection(geom)

        if not inter_geom.is_empty:
            intersecting_geoms.append(tile.intersection(geom))

    return intersecting_geoms


def _get_rounded_bounding_box(geom, width):
    """
    Round bounding box to divide evenly into width x width tiles from plane origin
    """
    return (
        geom.bounds[0] - (geom.bounds[0] % width),
        geom.bounds[1] - (geom.bounds[1] % width),
        geom.bounds[2] + (-geom.bounds[2] % width),
        geom.bounds[3] + (-geom.bounds[3] % width),
    )
