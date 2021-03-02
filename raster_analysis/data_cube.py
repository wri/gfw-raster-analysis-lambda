from datetime import datetime
from typing import List, Optional, Dict, Tuple, Set

from pydantic import BaseModel

import numpy as np
from numpy import ndarray
from rasterio.transform import Affine, from_bounds, xy
import concurrent.futures
from shapely.geometry import Polygon
from aws_xray_sdk.core import xray_recorder

from raster_analysis.io import mask_geom_on_raster
from raster_analysis.geodesy import get_area
from raster_analysis.utils import get_linear_index
from raster_analysis.globals import LOGGER, WINDOW_SIZE, BasePolygon
from raster_analysis.query import Query, LayerInfo
from raster_analysis.layer.window import get_window, Window, ResultValues


class DataCube(BaseModel):
    windows: Dict[LayerInfo, Window]
    shifted_affine: Affine
    mean_area: float
    mask: ndarray

    def __init__(
        self,
        geom: BasePolygon,
        tile: Polygon,
        layers: Set[LayerInfo],
    ):
        self.mean_area = get_area(tile.centroid.y) / 10000
        self.windows = self._get_windows(layers, tile)
        self.shifted_affine: Affine = from_bounds(
            *tile.bounds, WINDOW_SIZE, WINDOW_SIZE
        )

        self.mask = mask_geom_on_raster(
            np.ones((WINDOW_SIZE, WINDOW_SIZE)), self.shifted_affine, geom
        )

    def _get_windows(self, layers: Set[LayerInfo], tile):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its URL
            futures = {
                executor.submit(get_window, layer, tile): layer
                for layer in layers
            }

            return self._get_window_results(futures)

    @staticmethod
    def _get_window_results(futures):
        windows = {}
        for future in concurrent.futures.as_completed(futures):
            layer = futures[future]

            try:
                windows[layer] = future.result()
            except Exception as e:
                LOGGER.exception(f"Exception while reading window for layer {layer}")
                raise e

        return windows

