from shapely.geometry import mapping, box
import logging
import os
from datetime import date
import pandas as pd
from aws_xray_sdk.core import xray_recorder
from raster_analysis.boto import lambda_client
from raster_analysis.results_store import AnalysisResultsStore
from raster_analysis.globals import LOGGER


@xray_recorder.capture("Merge Tiled Geometry Results")
def merge_tile_results(tile_results, groupby_columns):
    if not groupby_columns:
        dataframes = [pd.DataFrame(result, index=[0]) for result in tile_results]
        merged_df: pd.DataFrame = pd.concat(dataframes)
        return merged_df.sum().to_dict()

    dataframes = [pd.DataFrame(result) for result in tile_results]
    merged_df: pd.DataFrame = pd.concat(dataframes)

    grouped_df: pd.DataFrame = merged_df.groupby(groupby_columns).sum()
    result_df: pd.DataFrame = grouped_df.sort_values(groupby_columns).reset_index()

    # convert ordinal dates to readable dates
    for col in groupby_columns:
        if "__date" in col:
            result_df[col] = result_df[col].apply(
                lambda val: date.fromordinal(val).strftime("%Y-%m-%d")
            )
        elif "__isoweek" in col:
            result_df[col.replace("__isoweek", "__year")] = result_df[col].apply(
                lambda val: date.fromordinal(val).isocalendar()[0]
            )
            result_df[col] = result_df[col].apply(
                lambda val: date.fromordinal(val).isocalendar()[1]
            )

    return result_df.to_dict(orient="records")


def process_tiled_geoms(tiles, geoprocessing_params, request_id, fanout_num):
    geom_count = len(tiles)
    geoprocessing_params["analysis_id"] = request_id
    LOGGER.info(f"Processing {geom_count} tiles")

    tile_geojsons = [mapping(tile) for tile in tiles]
    tile_chunks = [
        tile_geojsons[x : x + fanout_num]
        for x in range(0, len(tile_geojsons), fanout_num)
    ]
    fanout_lambda = os.environ["FANOUT_LAMBDA_NAME"]

    from raster_analysis.boto import invoke_lambda

    for chunk in tile_chunks:
        event = {"payload": geoprocessing_params, "tiles": chunk}
        invoke_lambda(event, fanout_lambda, lambda_client())

    LOGGER.info(f"Geom count: {geom_count}")
    results_store = AnalysisResultsStore(request_id)
    results = results_store.wait_for_results(geom_count)

    return results


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
