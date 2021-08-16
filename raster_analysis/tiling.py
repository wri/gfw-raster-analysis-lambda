import json
import sys
from copy import deepcopy
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Tuple
from hashlib import md5

from numpy import ceil
from pandas import DataFrame
from shapely.geometry import Polygon, box, mapping, shape

from raster_analysis.boto import invoke_lambda, lambda_client, dynamodb_client
from raster_analysis.data_environment import DataEnvironment
from raster_analysis.geometry import encode_geometry
from raster_analysis.globals import (
    FANOUT_LAMBDA_NAME,
    FANOUT_NUM,
    LAMBDA_ASYNC_PAYLOAD_LIMIT_BYTES,
    LOGGER,
    RASTER_ANALYSIS_LAMBDA_NAME,
    TILED_STATUS_TABLE_NAME,
    BasePolygon,
    Numeric,
)
from raster_analysis.query import Function, Query
from raster_analysis.results_store import AnalysisResultsStore, ResultStatus


class AnalysisTiler:
    def __init__(
        self,
        raw_query: str,
        raw_geom: Dict[str, Any],
        request_id: str,
        data_environment: DataEnvironment,
    ):
        self.raw_query: str = raw_query
        self.query: Query = Query(raw_query, data_environment)
        self.data_environment: DataEnvironment = data_environment

        self.raw_geom: Dict[str, Any] = raw_geom
        self.geom: BasePolygon = shape(raw_geom).buffer(0)
        self.grid = self.query.get_minimum_grid()

        self.request_id: str = request_id
        self.results: DataFrame = None

    def execute(self) -> None:
        self.results = self._execute_tiles()

        if self.results.size > 0:
            self.results = self._postprocess_results(self.results)

    def result_as_csv(self) -> str:
        if self.results.size > 0:
            buffer = StringIO()
            self.results.to_csv(buffer, index=False, float_format="%.5f")
            return buffer.getvalue()
        else:
            return ""

    def result_as_dict(self) -> Dict[str, Any]:
        return self.results.to_dict(orient="records")

    def _postprocess_results(self, results):
        """Decode results from pixel values if necessary for layer, apply any
        selector functions, and re-apply any group by's based on final
        values."""
        group_columns = self.query.get_group_columns()

        for selector in self.query.get_result_selectors():
            results[selector.layer] = self.data_environment.decode_layer(
                selector.layer, results[selector.layer]
            )

            if selector.function:
                # TODO put this out to a special processor
                layer = self.data_environment.get_layer(selector.layer)
                if selector.function == Function.isoweek:
                    years, isoweeks = zip(
                        *map(
                            lambda val: datetime.strptime(
                                val, "%Y-%m-%d"
                            ).isocalendar()[:2],
                            results[layer.name],
                        )
                    )

                    isoweek_col_name = layer.name.replace("date", "isoweek")
                    year_col_name = layer.name.replace("date", "year")

                    results[year_col_name] = years
                    results[isoweek_col_name] = isoweeks
                    del results[layer.name]

                    if group_columns:
                        group_columns.append(year_col_name)
                        group_columns.append(isoweek_col_name)
                        group_columns.remove(layer.name)

        if group_columns:
            grouped_df = results.groupby(group_columns).sum()
            return grouped_df.sort_values(group_columns).reset_index()
        elif self.query.aggregates:
            df = results.sum()

            # convert back to single row DF instead of Series
            df = DataFrame([df.values], columns=df.keys().values)
            return df
        else:
            return results

    def _execute_tiles(self) -> DataFrame:
        tiles_for_lambda = self._get_tiles(self.grid.tile_degrees)
        payload: Dict[str, Any] = {
            "analysis_id": self.request_id,
            "query": self.raw_query,
            "environment": self.data_environment.dict()["layers"],
        }

        if sys.getsizeof(json.dumps(self.raw_geom)) > LAMBDA_ASYNC_PAYLOAD_LIMIT_BYTES:
            payload["encoded_geometry"] = encode_geometry(self.geom)
        else:
            payload["geometry"] = self.raw_geom

        geom_count = len(tiles_for_lambda)

        LOGGER.info(f"Processing {geom_count} tiles")

        if geom_count <= FANOUT_NUM:
            for tile in tiles_for_lambda:
                tile_payload = deepcopy(payload)
                tile_id = self._get_cache_key(tile)
                tile_payload["tile_result_id"] = tile_id
                tile_payload["tile"] = mapping(tile)
                invoke_lambda(
                    tile_payload, RASTER_ANALYSIS_LAMBDA_NAME, lambda_client()
                )
        else:
            tile_geojsons = [
                (self._get_cache_key(tile), mapping(tile)) for tile in tiles_for_lambda
            ]
            tile_chunks = [
                tile_geojsons[x : x + FANOUT_NUM]
                for x in range(0, len(tile_geojsons), FANOUT_NUM)
            ]

            for chunk in tile_chunks:
                event = {"payload": payload, "tiles": chunk}
                invoke_lambda(event, FANOUT_LAMBDA_NAME, lambda_client())


        results_store = AnalysisResultsStore(self.request_id)
        all_tile_ids = [
            self._get_cache_key(tile) for tile in
            self._get_tiles(self.grid.tile_degrees, include_cached=True)
        ]
        lambda_tile_ids = [self._get_cache_key(tile) for tile in tiles_for_lambda]
        LOGGER.info(f"Geom count: going to lambda: {geom_count}, fetched from catch: {len(all_tile_ids) - geom_count}")

        results = results_store.wait_for_results(lambda_tile_ids, all_tile_ids)

        return results

    def _get_tiles(self, width: Numeric, include_cached=False) -> List[Polygon]:
        """Get width x width tile geometries over the extent of the
        geometry snapped to global tile grid of size width."""
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
                include_tile = self.geom.intersects(tile)
                if not include_cached:
                    include_tile = bool(include_tile * ~self._is_cached(tile))

                if include_tile:
                    tiles.append(tile)

        return tiles

    @staticmethod
    def _get_rounded_bounding_box(
        geom: BasePolygon, width: Numeric
    ) -> Tuple[int, int, int, int]:
        """Round bounding box to divide evenly into width x width tiles from
        plane origin."""
        return (
            geom.bounds[0] - (geom.bounds[0] % width),
            geom.bounds[1] - (geom.bounds[1] % width),
            geom.bounds[2] + (-geom.bounds[2] % width),
            geom.bounds[3] + (-geom.bounds[3] % width),
        )


    def _get_cache_key(self, tile: Polygon) -> str:
        "Create md5 has for tile-geom_overlap-query result"
        geom_tile_intersection = tile.intersection(self.geom)
        key = f"{self.raw_query}-{tile.wkt}-{geom_tile_intersection.wkt}"
        

        return md5(key.encode()).hexdigest()


    def _is_cached(self, tile: Polygon) -> bool:
        """Check if query result for the tile-geometry overlap area is in cache"""
        cache_key = self._get_cache_key(tile)

        tile_response = dynamodb_client().query(
            ExpressionAttributeValues={":id": {"S": cache_key}, },
            KeyConditionExpression="tile_id = :id",
            TableName=TILED_STATUS_TABLE_NAME,
        )

        return (
            len(tile_response["Items"]) > 0 and
            tile_response["Items"][0]["status"]["S"] == ResultStatus.success
        )
