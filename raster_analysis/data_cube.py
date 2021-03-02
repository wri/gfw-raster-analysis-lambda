from datetime import datetime
from typing import List, Optional, Dict, Tuple, Set

from pydantic import BaseModel

import numpy as np
from numpy import ndarray
from rasterio import features
from rasterio.transform import Affine, from_bounds, xy
import concurrent.futures
from shapely.geometry import Polygon

from raster_analysis.geodesy import get_area
from raster_analysis.globals import LOGGER, WINDOW_SIZE, BasePolygon
from raster_analysis.query import LayerInfo
from raster_analysis.window import Window


class DataCube:
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

        self.mask = self._mask_geom_on_raster(
            np.ones((WINDOW_SIZE, WINDOW_SIZE)), self.shifted_affine, geom
        )

    def _get_windows(self, layers: Set[LayerInfo], tile):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its URL
            futures = {
                executor.submit(Window, layer, tile): layer
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


    @staticmethod
    def _mask_geom_on_raster(
            raster_data: np.ndarray, shifted_affine: Affine, geom: BasePolygon
    ) -> np.ndarray:
        """"
        For a given polygon, returns a numpy masked array with the intersecting
        values of the raster at `raster_path` unmasked, all non-intersecting
        cells are masked.  This assumes that the input geometry is in the same
        SRS as the raster.  Currently only reads from a single band.

        Args:
            geom (Shapely Geometry): A polygon in the same SRS as `raster_path`
                which will define the area of the raster to mask.

            raster_path (string): A local file path to a geographic raster
                containing values to extract.

        Returns
           Numpy masked array of source raster, cropped to the extent of the
           input geometry, with any modifications applied. Areas where the
           supplied geometry does not intersect are masked.

        """

        if raster_data.size > 0:
            # Create a numpy array to mask cells which don't intersect with the
            # polygon. Cells that intersect will have value of 1 (unmasked), the
            # rest are filled with 0s (masked)
            geom_mask = features.geometry_mask(
                [geom], out_shape=raster_data.shape, transform=shifted_affine, invert=True
            )

            # Mask the data array, with modifications applied, by the query polygon
            return geom_mask
        else:
            return np.array([])


