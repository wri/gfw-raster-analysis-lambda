from datetime import datetime
from typing import List
import numpy as np
import concurrent.futures

from raster_analysis.io import mask_geom_on_raster
from raster_analysis.geodesy import get_area
from raster_analysis.numpy_utils import get_linear_index
from raster_analysis.globals import LOGGER
from .window import get_window, WINDOW_SIZE


class DataCube:
    """
    A stack of raster windows for analysis
    """

    def __init__(
        self,
        geom,
        tile,
        group_layers: List[str],
        sum_layers: List[str],
        filter_layers: List[str],
        start_date: datetime,
        end_date: datetime,
    ):
        self.geom = geom
        self.mean_area = get_area(tile.centroid.y) / 10000

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

            def get_window_results(futures):
                windows = []
                for future in concurrent.futures.as_completed(futures):
                    layer = futures[future]

                    try:
                        windows.append(future.result())
                    except Exception:
                        LOGGER.exception(
                            f"Exception while reading window for layer {layer}"
                        )

                return windows

            self.group_windows = get_window_results(group_futures)
            self.sum_windows = get_window_results(sum_futures)
            self.filter_windows = get_window_results(filter_futures)

        self.data_windows = [
            w
            for w in self.group_windows + self.filter_windows + self.sum_windows
            if w.data is not None
        ]
        if len(self.data_windows) < 1:
            raise ValueError("No windows with data could be found")

        # TODO since I'm always getting a tile now, should I just precalc affine?
        self.shifted_affine = self.data_windows[0].shifted_affine

        # generate filter and then clear those rasters to save memory
        self.filter = mask_geom_on_raster(
            np.ones((WINDOW_SIZE, WINDOW_SIZE)), self.shifted_affine, self.geom
        )
        for window in self.filter_windows:
            self.filter *= window.data.astype(np.bool)
            window.clear()

        # we only care about areas where the group has data, so also add those to the filter
        for window in self.group_windows:
            try:
                self.filter *= window.data.astype(np.bool)
            except ValueError as e:
                raise e

        # generate linear index and then clear group rasters to save data
        if self.group_windows:
            group_cols = [np.ravel(window.data) for window in self.group_windows]
            self.index_dims = [col.max() + 1 for col in group_cols]
            self.linear_index = get_linear_index(
                group_cols, self.index_dims, self.filter
            )

            for window in self.group_windows:
                window.clear()

    def calculate(self):
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

        return dict([window.result for window in self.group_windows + self.sum_windows])
