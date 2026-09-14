import json
import math
import sys
from copy import deepcopy
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Tuple, Optional, Protocol

from pandas import DataFrame
from shapely.geometry import Polygon, box, mapping, shape
from shapely.prepared import prep

from raster_analysis.boto import invoke_lambda, lambda_client
from raster_analysis.data_environment import DataEnvironment
from raster_analysis.geometry import encode_geometry
from raster_analysis.globals import (
    FANOUT_LAMBDA_NAME,
    FANOUT_NUM,
    LAMBDA_ASYNC_PAYLOAD_LIMIT_BYTES,
    LOGGER,
    RASTER_ANALYSIS_LAMBDA_NAME,
    BasePolygon,
    Numeric,
)
from raster_analysis.query import Function, Query, Sort
from raster_analysis.results_store import AnalysisResultsStore, ResultStatus


class Invoker(Protocol):
    def invoke(self, payload: Dict[str, Any], function_name: str) -> None:
        pass

class LambdaInvoker(Invoker):
    def invoke(self, payload: Dict[str, Any], function_name: str) -> None:
        invoke_lambda(payload, function_name, lambda_client())


class AnalysisTiler:
    def __init__(
        self,
        raw_query: str,
        raw_geom: Dict[str, Any],
        request_id: str,
        data_environment: DataEnvironment,
        *,
        results_store: Optional[AnalysisResultsStore] = None,
        invoker: Optional[Invoker] = None,
    ):
        self.raw_query: str = raw_query
        self.query: Query = Query(raw_query, data_environment)
        self.data_environment: DataEnvironment = data_environment

        self.raw_geom: Dict[str, Any] = raw_geom
        self.geom: BasePolygon = shape(raw_geom).buffer(0)
        self.grid = self.query.get_minimum_grid()

        self.request_id: str = request_id
        self.results: DataFrame = None

        self.results_store = results_store or AnalysisResultsStore()
        self.invoker = invoker or LambdaInvoker()

    def execute(self) -> DataFrame:
        self.results = self._execute_tiles()

        if self.results.size > 0:
            self.results = self._postprocess_results(self.results)
        return self.results

    def result_as_csv(self) -> str:
        if self.results.size > 0:
            buffer = StringIO()
            self.results.to_csv(buffer, index=False, float_format="%.5f")
            return buffer.getvalue()
        else:
            return ""

    def result_as_dict(self) -> Dict[str, Any]:
        results = self.results.to_dict(orient="records")

        # deal with pandas converting ints to floats
        for result in results:
            for key, value in result.items():
                if isinstance(value, float) and value.is_integer():
                    result[key] = int(value)

        return results

    def _postprocess_results(self, results):
        """Decode results from pixel values if necessary for layer, apply any
        selector functions, and re-apply any group by's based on final
        values."""
        group_columns = self.query.get_group_columns()
        order_by_columns = self.query.get_order_by_columns()
        selectors = self.query.get_result_selectors()

        for selector in selectors:
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
            results = results.groupby(group_columns).sum().reset_index()
        elif self.query.aggregates:
            results = results.sum()

            # convert back to single row DF instead of Series
            results = DataFrame([results.values], columns=results.keys().values)

        if order_by_columns:
            is_asc = self.query.sort == Sort.asc
            results = results.sort_values(order_by_columns, ascending=is_asc)

        if self.query.limit:
            results = results.head(self.query.limit)

        # change column names for AS statements
        aliases = {
            col.layer: col.alias
            for col in selectors + self.query.aggregates
            if col.alias
        }
        results = results.rename(columns=aliases)

        return results

    def _execute_tiles(self) -> DataFrame:
        payload: Dict[str, Any] = {
            "query": self.raw_query,
            "environment": self.data_environment.dict(),
        }

        payload["geometry"] = self.raw_geom
        if sys.getsizeof(json.dumps(payload)) > LAMBDA_ASYNC_PAYLOAD_LIMIT_BYTES:
            # if payload would be too big, compress geometry
            geom = shape(payload.pop("geometry"))
            payload["encoded_geometry"] = encode_geometry(geom)

        tiles = self._get_tiles(self.grid.tile_degrees)

        results_store = self.results_store

        # Compute each cache key exactly once — get_cache_key does a Shapely
        # intersection + WKT serialization + MD5 hash per tile, so repeated
        # calls for the same tile are wasteful.
        tile_key_map: Dict[int, str] = {
            id(tile): results_store.get_cache_key(tile, self.geom, self.raw_query)
            for tile in tiles
        }
        tile_keys = [tile_key_map[id(tile)] for tile in tiles]

        cached_tile_keys = {
            status["tile_id"]["S"]
            for status in results_store.get_statuses(
                tile_keys, status_filter=ResultStatus.success
            )
        }
        tiles_for_lambda = [
            tile for tile in tiles if tile_key_map[id(tile)] not in cached_tile_keys
        ]
        cache_keys_for_lambda = [tile_key_map[id(tile)] for tile in tiles_for_lambda]
        geom_count = len(tiles_for_lambda)

        LOGGER.info(f"Processing {geom_count} tiles")

        if geom_count <= FANOUT_NUM:
            for tile in tiles_for_lambda:
                tile_payload = deepcopy(payload)
                tile_payload["cache_id"] = tile_key_map[id(tile)]
                tile_payload["tile"] = mapping(tile)
                self.invoker.invoke(
                    tile_payload, RASTER_ANALYSIS_LAMBDA_NAME
                )
        else:
            tile_geojsons = [
                (tile_key_map[id(tile)], mapping(tile))
                for tile in tiles
            ]
            tile_chunks = [
                tile_geojsons[x : x + FANOUT_NUM]
                for x in range(0, len(tile_geojsons), FANOUT_NUM)
            ]

            for chunk in tile_chunks:
                event = {"payload": payload, "tiles": chunk}
                self.invoker.invoke(event, FANOUT_LAMBDA_NAME)

        LOGGER.info(
            f"Geom count: going to lambda: {geom_count}, fetched from cache: {len(tile_keys) - geom_count}"
        )

        results = results_store.wait_for_results(cache_keys_for_lambda, tile_keys)

        return results

    def _get_tiles(self, width: Numeric) -> List[Polygon]:
        """Get width x width tile geometries over the extent of the geometry
        snapped to global tile grid of size width."""
        min_x, min_y, max_x, max_y = self._get_rounded_bounding_box(self.geom, width)

        # Number of tiles in each dimension (ceil avoids dropping edge tiles)
        nx = int(math.ceil((max_x - min_x) / width))
        ny = int(math.ceil((max_y - min_y) / width))

        gprep = prep(self.geom)
        tiles: List[Polygon] = []

        # Localize for speed in Python loops
        bx = box
        base_x = min_x
        base_y = min_y
        w = width

        for i in range(nx):
            x0 = (i * w) + base_x
            x1 = x0 + w
            for j in range(ny):
                y0 = (j * w) + base_y
                y1 = y0 + w
                tile = bx(x0, y0, x1, y1)
                if gprep.intersects(tile):
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
