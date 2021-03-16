import json
import sys
from io import StringIO
from typing import Dict, List, Any, Tuple
from copy import deepcopy

from pandas import DataFrame
from shapely.geometry import Polygon, mapping, box, shape
from aws_xray_sdk.core import xray_recorder

from raster_analysis.boto import lambda_client, invoke_lambda
from raster_analysis.geometry import encode_geometry
from raster_analysis.query import Query
from raster_analysis.results_store import AnalysisResultsStore
from raster_analysis.globals import (
    LOGGER,
    FANOUT_LAMBDA_NAME,
    RASTER_ANALYSIS_LAMBDA_NAME,
    BasePolygon,
    Numeric,
    LAMBDA_ASYNC_PAYLOAD_LIMIT_BYTES,
    TILE_WIDTH,
    FANOUT_NUM,
)


class AnalysisTiler:
    def __init__(self, raw_query: str, raw_geom: Dict[str, Any], request_id: str):
        self.raw_query: str = raw_query
        self.query: Query = Query.parse_query(raw_query)

        self.raw_geom: Dict[str, Any] = raw_geom
        self.geom: BasePolygon = shape(raw_geom)

        self.request_id: str = request_id
        self.results: DataFrame = None

    def execute(self) -> None:
        self.results = self._execute_tiles()

        if self.results.size > 0:
            self.results = self._decode_and_group_results(self.results)

    def result_as_csv(self) -> StringIO:
        if self.results.size > 0:
            buffer = StringIO()
            self.results.to_csv(buffer, index=False, float_format="%.5f")
            return buffer
        else:
            return StringIO()

    def _decode_and_group_results(self, results):
        group_columns = self.query.get_group_columns()

        for layer in self.query.get_result_layers():
            if layer.decoder and layer.layer in results:
                decoded_columns = layer.decoder(results[layer.layer])
                del results[layer.layer]

                for name, series in decoded_columns.items():
                    results[name] = series

                if layer.layer in group_columns:
                    group_columns.remove(layer.layer)
                    group_columns += [name for name in decoded_columns.keys()]

        if group_columns:
            grouped_df = results.groupby(group_columns).sum()
            return grouped_df.sort_values(group_columns).reset_index()
        elif self.query.aggregates:
            # pandas does weird things when you sum the whole DF
            df = results.sum().reset_index()
            df = df.rename(columns=df["index"])
            df = df.drop(columns=["index"])
            return df
        else:
            return results

    @xray_recorder.capture("Process Tiles")
    def _execute_tiles(self) -> DataFrame:
        tiles = self._get_tiles(TILE_WIDTH)
        payload: Dict[str, Any] = {
            "analysis_id": self.request_id,
            "query": self.raw_query,
        }

        if sys.getsizeof(json.dumps(self.raw_geom)) > LAMBDA_ASYNC_PAYLOAD_LIMIT_BYTES:
            payload["encoded_geometry"] = encode_geometry(self.geom)
        else:
            payload["geometry"] = self.raw_geom

        geom_count = len(tiles)

        LOGGER.info(f"Processing {geom_count} tiles")

        if geom_count <= FANOUT_NUM:
            for tile in tiles:
                tile_payload = deepcopy(payload)
                tile_payload["tile"] = mapping(tile)
                invoke_lambda(
                    tile_payload, RASTER_ANALYSIS_LAMBDA_NAME, lambda_client()
                )
        else:
            tile_geojsons = [mapping(tile) for tile in tiles]
            tile_chunks = [
                tile_geojsons[x : x + FANOUT_NUM]
                for x in range(0, len(tile_geojsons), FANOUT_NUM)
            ]

            for chunk in tile_chunks:
                event = {"payload": payload, "tiles": chunk}
                invoke_lambda(event, FANOUT_LAMBDA_NAME, lambda_client())

        LOGGER.info(f"Geom count: {geom_count}")
        results_store = AnalysisResultsStore(self.request_id)
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
    def _get_rounded_bounding_box(
        geom: BasePolygon, width: Numeric
    ) -> Tuple[int, int, int, int]:
        """
        Round bounding box to divide evenly into width x width tiles from plane origin
        """
        return (
            geom.bounds[0] - (geom.bounds[0] % width),
            geom.bounds[1] - (geom.bounds[1] % width),
            geom.bounds[2] + (-geom.bounds[2] % width),
            geom.bounds[3] + (-geom.bounds[3] % width),
        )
