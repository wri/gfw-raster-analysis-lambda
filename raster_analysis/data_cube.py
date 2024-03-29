import concurrent.futures
from typing import List

import numpy as np
from rasterio import features
from rasterio.transform import Affine, from_bounds
from shapely.geometry import Polygon

from raster_analysis.data_environment import DataEnvironment, SourceLayer
from raster_analysis.geodesy import get_area
from raster_analysis.globals import LOGGER, BasePolygon
from raster_analysis.query import Query
from raster_analysis.window import DerivedWindow, SourceWindow


class DataCube:
    def __init__(self, geom: BasePolygon, tile: Polygon, query: Query):
        self.grid = query.get_minimum_grid()
        self.mean_area = get_area(tile.centroid.y, self.grid.get_pixel_width()) / 10000

        source_layers = query.get_source_layers()
        layer_names = query.get_layer_names()
        self.windows = self._get_windows(source_layers, tile, query.data_environment)

        for layer in query.get_derived_layers():
            self.windows[layer.name] = DerivedWindow(
                layer, self.windows[layer.source_layer], self.mean_area
            )

        for layer in source_layers:
            if layer.name not in layer_names and layer.name in self.windows:
                # free up memory, since this means the source layer was only required
                # for a derived layer
                del self.windows[layer.name]

        tile_width = self.grid.get_tile_width()
        self.shifted_affine: Affine = from_bounds(*tile.bounds, tile_width, tile_width)

        self.mask = self._mask_geom_on_raster(
            np.ones((tile_width, tile_width)), self.shifted_affine, geom
        )

    def _get_windows(
        self,
        layers: List[SourceLayer],
        tile: BasePolygon,
        data_environment: DataEnvironment,
    ):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its URL
            futures = {
                executor.submit(
                    SourceWindow, layer, tile, self.grid, data_environment
                ): layer
                for layer in layers
            }

            return self._get_window_results(futures)

    @staticmethod
    def _get_window_results(futures):
        windows = {}
        for future in concurrent.futures.as_completed(futures):
            layer = futures[future]

            try:
                windows[layer.name] = future.result()
            except Exception as e:
                LOGGER.exception(f"Exception while reading window for layer {layer}")
                raise e

        return windows

    @staticmethod
    def _mask_geom_on_raster(
        raster_data: np.ndarray, shifted_affine: Affine, geom: BasePolygon
    ) -> np.ndarray:
        """For a given polygon, returns a numpy masked array with the
        intersecting values of the raster_data, all non- intersecting cells are
        masked.  This assumes that the input geometry is in the same SRS as the
        raster.

        Args:
            raster_data (ndarray): ndarray of raster data to be masked onto
            shifted_affine (Affine): the transform used to map the vector geometry to the raster
            geom (Shapely Geometry): A polygon in the same SRS as `raster_path`
                which will define the area of the raster to mask.

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
