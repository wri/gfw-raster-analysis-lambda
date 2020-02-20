from shapely.geometry import mapping, box

import boto3
import json
import logging
import os

import threading
import queue

from copy import deepcopy

import numpy as np

from raster_analysis.exceptions import RasterAnalysisException
from aws_xray_sdk.core import xray_recorder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RASTER_ANALYSIS_LAMBDA_NAME = os.environ["RASTER_ANALYSIS_LAMBDA_NAME"]


@xray_recorder.capture("Merge Tiled Geometry Results")
def merge_tile_results(tile_results, groupby_columns):
    concatted_tile_results = concat_tile_results(tile_results)

    groupby_results = np.array([concatted_tile_results[col] for col in groupby_columns])
    column_maxes = [col.max() + 1 for col in groupby_results]
    linear_indices = np.ravel_multi_index(groupby_results, column_maxes).astype(
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

    for col, arr in merged_tile_results.items():
        merged_tile_results[col] = arr.tolist()

    return merged_tile_results


def concat_tile_results(tile_results):
    result = deepcopy(tile_results[0])

    for col in result.keys():
        for table in tile_results[1:]:
            result[col] += table[col]

    return result


@xray_recorder.capture("Process Tiled Geometries")
def process_tiled_geoms(tiled_geoms, geoprocessing_params):
    execution_threads = []
    result_queue = queue.Queue()
    error_queue = queue.Queue()

    for geom in tiled_geoms:
        payload = geoprocessing_params.copy()
        payload["geometry"] = mapping(geom)

        execution_thread = threading.Thread(
            target=raster_analysis_worker, args=(payload, result_queue, error_queue)
        )
        execution_thread.start()
        execution_threads.append(execution_thread)

    for execution_thread in execution_threads:
        execution_thread.join()

    errors = [error_queue.get() for _ in range(error_queue.qsize())]
    if errors:
        logger.error("Error in raster analyses lambda. Check logs.")
        raise RasterAnalysisException("Error in raster analyses lambda. Check logs.")

    results = [result_queue.get() for _ in range(result_queue.qsize())]

    # get key to check if result has no rows
    result_random_key = list(results[0].keys())[0]
    nonempty_results = list(filter(lambda result: result[result_random_key], results))

    return nonempty_results


def raster_analysis_worker(payload, result_queue, error_queue):
    try:
        result_table = run_raster_analysis(payload)
        result_queue.put(result_table)
    except Exception as e:
        logger.error(e)
        error_queue.put(True)


def run_raster_analysis(payload):
    lambda_client = boto3.Session().client("lambda")

    lambda_response = lambda_client.invoke(
        FunctionName=RASTER_ANALYSIS_LAMBDA_NAME,
        InvocationType="RequestResponse",
        Payload=bytes(json.dumps(payload), "utf-8"),
    )

    response = json.loads(lambda_response["Payload"].read())
    result = json.loads(response["body"])

    if response["statusCode"] == 200:
        return result
    else:
        logger.error(f"Status code: {response['status_code']}\nContent: {result}")
        raise Exception(f"Status code: {response['status_code']}")


@xray_recorder.capture("Get Tiles")
def get_tiles(geom, width):
    """
    Get width x width tile geometries over the extent of the gseometry
    """
    min_x, min_y, max_x, max_y = _get_rounded_bounding_box(geom, width)
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
