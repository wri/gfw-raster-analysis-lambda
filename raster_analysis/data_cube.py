from typing import List

import numpy as np
from rasterio import features
from rasterio.transform import Affine, from_bounds
import concurrent.futures
from shapely.geometry import Polygon

from raster_analysis.layer import Layer, Grid
from raster_analysis.geodesy import get_area
from raster_analysis.globals import LOGGER, WINDOW_SIZE, BasePolygon
from raster_analysis.query import Query
from raster_analysis.window import Window


class DataCube:
    def __init__(self, geom: BasePolygon, tile: Polygon, query: Query):
        self.grid = query.get_minimum_grid()
        self.mean_area = get_area(tile.centroid.y, self.grid.get_pixel_width()) / 10000
        self.windows = self._get_windows(query.get_real_layers(), tile)
        self._expand_encoded_layers(query)

        tile_width = self.grid.get_tile_width()
        self.shifted_affine: Affine = from_bounds(*tile.bounds, tile_width, tile_width)

        self.mask = self._mask_geom_on_raster(
            np.ones((tile_width, tile_width)), self.shifted_affine, geom
        )

    def _expand_encoded_layers(self, query: Query):
        layers = query.get_layers()

        # TODO should be only reading each layer once
        for layer in layers:
            if layer.alias in [
                "umd_glad_alerts__confidence",
                "umd_glad_landsat_alerts__confidence",
                "gfw_radd_alerts__confidence",
            ]:
                self.windows[layer].data = np.floor(
                    self.windows[layer].data / 10000
                ).astype(dtype=np.uint8)
            elif layer.layer in [
                "umd_glad_alerts__date",
                "umd_glad_landsat_alerts__date",
                "gfw_radd_alerts__date_conf",
            ]:
                self.windows[layer].data = self.windows[layer].data % 10000

    def _get_windows(self, layers: List[Layer], tile):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its URL
            futures = {
                executor.submit(Window, layer, tile, self.grid): layer
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
                [geom],
                out_shape=raster_data.shape,
                transform=shifted_affine,
                invert=True,
            )

            # Mask the data array, with modifications applied, by the query polygon
            return geom_mask
        else:
            return np.array([])
