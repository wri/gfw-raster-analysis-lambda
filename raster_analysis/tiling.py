from datetime import date
from typing import Dict, List, Any, Tuple
from copy import deepcopy

import pandas as pd
from pandas import DataFrame
from shapely.geometry import Polygon, mapping, box
from aws_xray_sdk.core import xray_recorder

from raster_analysis.boto import lambda_client, invoke_lambda
from raster_analysis.data_lake import LAYERS
from raster_analysis.query import Query
from raster_analysis.results_store import AnalysisResultsStore
from raster_analysis.globals import (
    LOGGER,
    FANOUT_LAMBDA_NAME,
    RASTER_ANALYSIS_LAMBDA_NAME,
    ResultValue,
    BasePolygon,
    Numeric,
)


class AnalysisTiler:
    def __init__(self, query: Query, geom: BasePolygon):
        self.query = query
        self.geom = geom

    def execute(self):
        self._process_tiles()

    def _group_results(self, results, groupby_columns):
        grouped_df: pd.DataFrame = results.groupby(self.groupby_columns).sum()
        return grouped_df.sort_values(groupby_columns).reset_index()

    def _decode_values(self):
        for layer in set([self.query.selectors + self.query.groups]):
            if layer.value_decoder:
                decoded_columns = layer.value_decoder(self.results[layer.layer])
                del self.results[layer.layer]
                for name, series in decoded_columns:
                    self.results[name] = series

    @xray_recorder.capture("Process Tiles")
    def _process_tiles(
            self,
            tiles: List[Polygon],
            geoprocessing_params: Dict[str, Any],
            request_id: str,
            fanout_num: int,
    ) -> DataFrame:
        geom_count = len(tiles)
        geoprocessing_params["analysis_id"] = request_id
        LOGGER.info(f"Processing {geom_count} tiles")

        if geom_count <= fanout_num:
            for tile in tiles:
                tile_params = deepcopy(geoprocessing_params)
                tile_params["tile"] = mapping(tile)
                invoke_lambda(tile_params, RASTER_ANALYSIS_LAMBDA_NAME, lambda_client())
        else:
            tile_geojsons = [mapping(tile) for tile in tiles]
            tile_chunks = [
                tile_geojsons[x: x + fanout_num]
                for x in range(0, len(tile_geojsons), fanout_num)
            ]

            for chunk in tile_chunks:
                event = {"payload": geoprocessing_params, "tiles": chunk}
                invoke_lambda(event, FANOUT_LAMBDA_NAME, lambda_client())

        LOGGER.info(f"Geom count: {geom_count}")
        results_store = AnalysisResultsStore(request_id)
        results = results_store.wait_for_results(geom_count)

        return results

    def _get_tiles(self, width: Numeric) -> List[Polygon]:
        """
        Get width x width tile geometries over the extent of the geometry
        """
        min_x, min_y, max_x, max_y = self._get_rounded_bounding_box(self.geom, width)
        tiles = []

        for i in range(0, int((max_x - min_x) / width)):
            for j in range(0, int((max_y - min_y) / width)):
                tile = box(
                    (i * width) + min_x,
                    (j * width) + min_y,
                    ((i + 1) * width) + min_x,
                    ((j + 1) * width) + min_y,
                )

                if self.geom.intersects(tile):
                    tiles.append(tile)

        return tiles

    @staticmethod
    def _get_rounded_bounding_box(geom: BasePolygon, width: Numeric) -> Tuple[int, int, int, int]:
        """
        Round bounding box to divide evenly into width x width tiles from plane origin
        """
        return (
            geom.bounds[0] - (geom.bounds[0] % width),
            geom.bounds[1] - (geom.bounds[1] % width),
            geom.bounds[2] + (-geom.bounds[2] % width),
            geom.bounds[3] + (-geom.bounds[3] % width),
        )
