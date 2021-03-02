from datetime import date
from typing import Dict, List, Any
from copy import deepcopy

import pandas as pd
from pandas import DataFrame
from shapely.geometry import Polygon, mapping, box
from aws_xray_sdk.core import xray_recorder

from raster_analysis.boto import lambda_client, invoke_lambda
from raster_analysis.results_store import AnalysisResultsStore
from raster_analysis.globals import (
    LOGGER,
    FANOUT_LAMBDA_NAME,
    RASTER_ANALYSIS_LAMBDA_NAME,
    DATA_LAKE_LAYER_MANAGER,
    ResultValue,
    BasePolygon,
    Numeric,
)


class ResultsMerge:
    def __init__(self, tile_results: DataFrame, query: Query):
        self.results = tile_results
        self.groupby_columns = groupby_columns

    def group(self):
        grouped_df: pd.DataFrame = self.results.groupby(self.groupby_columns).sum()
        result_df: pd.DataFrame = grouped_df.sort_values(self.groupby_columns).reset_index()

    def decode_values(self):
        for layer in self.query.get_layers():
            encoding = DATA_LAKE_LAYER_MANAGER.layers[layer].encoding
            if encoding:
                self.results[layer.name_type] = self.results[layer.name_type].map(encoding)


@xray_recorder.capture("Merge Tiled Geometry Results")
def merge_tile_results(
    tile_results: DataFrame, groupby_columns: List[str]
) -> List[Dict[str, ResultValue]]:
    # TODO how to deal with this?
    # if not groupby_columns:
    #     dataframes = [pd.DataFrame(result, index=[0]) for result in tile_results]
    #     merged_df: pd.DataFrame = pd.concat(dataframes)
    #     return merged_df.sum().to_dict()
    grouped_df: pd.DataFrame = tile_results.groupby(groupby_columns).sum()
    result_df: pd.DataFrame = grouped_df.sort_values(groupby_columns).reset_index()

    # convert ordinal dates to readable dates
    for col in groupby_columns:
        if "__date" in col:
            result_df[col] = result_df[col].apply(
                lambda val: date.fromordinal(val).strftime("%Y-%m-%d")
            )
        elif "__isoweek" in col:
            result_df[col.replace("__isoweek", "__year")] = (
                result_df[col]
                .astype(int)
                .apply(lambda val: date.fromordinal(val).isocalendar()[0])
            )
            result_df[col] = (
                result_df[col]
                .astype(int)
                .apply(lambda val: date.fromordinal(val).isocalendar()[1])
            )

        # sometimes pandas makes int fields into floats - a group by field should never be a float
        if result_df[col].dtype == "float64":
            result_df[col] = result_df[col].astype("int64")

    return result_df.to_dict(orient="records")


@xray_recorder.capture("Process Tiles")
def process_tiled_geoms(
    tiles: List[Polygon],
    geoprocessing_params: Dict[str, Any],
    request_id: str,
    fanout_num: int,
):
    geom_count = len(tiles)
    geoprocessing_params["analysis_id"] = request_id
    LOGGER.info(f"Processing {geom_count} tiles")

    # if
    if geom_count <= fanout_num:
        for tile in tiles:
            tile_params = deepcopy(geoprocessing_params)
            tile_params["tile"] = mapping(tile)
            invoke_lambda(tile_params, RASTER_ANALYSIS_LAMBDA_NAME, lambda_client())
    else:
        tile_geojsons = [mapping(tile) for tile in tiles]
        tile_chunks = [
            tile_geojsons[x : x + fanout_num]
            for x in range(0, len(tile_geojsons), fanout_num)
        ]

        for chunk in tile_chunks:
            event = {"payload": geoprocessing_params, "tiles": chunk}
            invoke_lambda(event, FANOUT_LAMBDA_NAME, lambda_client())

    LOGGER.info(f"Geom count: {geom_count}")
    results_store = AnalysisResultsStore(request_id)
    results = results_store.wait_for_results(geom_count)

    return results


@xray_recorder.capture("Get Tiles")
def get_tiles(geom: BasePolygon, width: Numeric) -> List[Polygon]:
    """
    Get width x width tile geometries over the extent of the geometry
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


def _get_rounded_bounding_box(geom: BasePolygon, width: Numeric):
    """
    Round bounding box to divide evenly into width x width tiles from plane origin
    """
    return (
        geom.bounds[0] - (geom.bounds[0] % width),
        geom.bounds[1] - (geom.bounds[1] % width),
        geom.bounds[2] + (-geom.bounds[2] % width),
        geom.bounds[3] + (-geom.bounds[3] % width),
    )
