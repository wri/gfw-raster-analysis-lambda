from datetime import datetime
from typing import List, Optional, Dict, Tuple

import numpy as np
from rasterio.transform import Affine, from_bounds, xy
import concurrent.futures
from shapely.geometry import Polygon
from aws_xray_sdk.core import xray_recorder

from raster_analysis.io import mask_geom_on_raster
from raster_analysis.geodesy import get_area
from raster_analysis.utils import get_linear_index
from raster_analysis.globals import LOGGER, WINDOW_SIZE, BasePolygon
from .window import get_window, Window, ResultValues


class DataCube:
    """
    A stack of raster windows for analysis
    """

    def __init__(
        self,
        geom: BasePolygon,
        tile: Polygon,
        select_layers: List[str],
        where_layers: List[str],
        groupby_layers: List[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ):
        """
       Create a datacube of many raster layers at once, and compute basic SQL-like operations across them.

       :param geom: shapely geometry you want zonal statistics
       :param groups: layers whose values you want to group results by. Layer must be int, datetime or string. Equivalent to
               `select **groups from table group by **groups`
       :param sums: sum values of these layers based on group layers. Equivalent to sum aggregate function in SQL.
       :param filters: layers to mask input geometry with, where all no-data values are considered to be masked values.
       :param start_date: start date to filter datetime layers by
       :param end_date: end date to filter datetime layers by
       """
        self.geom: BasePolygon = geom
        self.mean_area: float = get_area(tile.centroid.y) / 10000

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its URL
            group_futures = {
                executor.submit(get_window, layer, tile, start_date, end_date): layer
                for layer in group_layers
            }
            sum_futures = {
                executor.submit(get_window, layer, tile, start_date, end_date): layer
                for layer in sum_layers
            }
            filter_futures = {
                executor.submit(get_window, layer, tile, start_date, end_date): layer
                for layer in filter_layers
            }

            self.group_windows: List[Window] = self._get_window_results(group_futures)
            self.sum_windows: List[Window] = self._get_window_results(sum_futures)
            self.filter_windows: List[Window] = self._get_window_results(filter_futures)

        self.shifted_affine: Affine = from_bounds(
            *tile.bounds, WINDOW_SIZE, WINDOW_SIZE
        )

        # generate filter and then clear those rasters to save memory
        self.filter: np.ndarray = mask_geom_on_raster(
            np.ones((WINDOW_SIZE, WINDOW_SIZE)), self.shifted_affine, self.geom
        )
        for window in self.filter_windows:
            self.filter *= window.data.astype(np.bool)
            window.clear()

        # we only care about areas where the group has data, so also add those to the filter
        # unless the window has a default value we care about
        for window in self.group_windows:
            if not window.has_default_value():
                self.filter *= window.data.astype(np.bool)

        # generate linear index and then clear group rasters to save data
        if self.group_windows:
            group_cols = [np.ravel(window.data) for window in self.group_windows]
            self.index_dims: List[int] = [col.max() + 1 for col in group_cols]
            self.linear_index: np.ndarray = get_linear_index(
                group_cols, self.index_dims, self.filter
            )

            for window in self.group_windows:
                window.clear()

    @xray_recorder.capture("Calculate")
    def calculate(self) -> Dict[str, ResultValues]:
        inverse_index = None
        counts = None

        if hasattr(self, "linear_index"):
            unique_values, inverse_index, counts = np.unique(
                self.linear_index, return_counts=True, return_inverse=True
            )
            unique_value_combinations = np.unravel_index(unique_values, self.index_dims)
            for window, result in zip(self.group_windows, unique_value_combinations):
                window.result = result

        for window in self.sum_windows:
            window.result = window.sum(
                self.mean_area, self.filter, inverse_index, counts
            )

        return {
            window.result_col_name: window.result
            for window in self.group_windows + self.sum_windows
        }

    def extract_coordinates(self) -> List[Tuple[float, float]]:
        if self.select_layer:
            points = self.filter * self.select_layer
            rows, cols = np.nonzero(points)
            lon_lat = zip(xy(self.shifted_affine, rows, cols))
            return lon_lat

    @staticmethod
    def _get_window_results(futures):
        windows = []
        for future in concurrent.futures.as_completed(futures):
            layer = futures[future]

            try:
                windows.append(future.result())
            except Exception as e:
                LOGGER.exception(f"Exception while reading window for layer {layer}")
                raise e

        return windows
