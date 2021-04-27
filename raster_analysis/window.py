# flake8: noqa
import logging
import traceback
from typing import Tuple

import numpy as np
import rasterio
import rasterio.windows as wd
from numpy import datetime64, floor, ndarray
from rasterio import DatasetReader
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from shapely.geometry import Polygon

from raster_analysis.data_environment import DataEnvironment, DerivedLayer, SourceLayer
from raster_analysis.globals import LOGGER, BasePolygon, Numeric
from raster_analysis.grid import Grid


class SourceWindow:
    def __init__(
        self,
        layer: SourceLayer,
        tile: Polygon,
        grid: Grid,
        data_environment: DataEnvironment,
    ):
        self.layer = layer
        self.tile = tile
        self.grid = grid
        self.source_uri = data_environment.get_source_uri(layer.name, tile)

        data, shifted_affine, no_data_value = self.read()

        self.data: ndarray = data
        self.shifted_affine: Affine = shifted_affine
        self.no_data_value: Numeric = no_data_value

    def read(self) -> Tuple[np.ndarray, Affine, Numeric]:
        data, shifted_affine, no_data_value = self._read_window_ignore_missing(
            self.source_uri, self.tile, self.grid
        )

        return data, shifted_affine, no_data_value

    def clear(self) -> None:
        """Clear internal data array to save memory."""
        self.data = []

    @staticmethod
    def _read_window(
        raster: str, geom: BasePolygon, grid: Grid, masked: bool = False
    ) -> Tuple[np.ndarray, Affine, Numeric]:
        """Read a chunk of the raster that contains the bounding box of the
        input geometry.

        This has memory implications if that rectangle is large. The
        affine transformation maps geom coordinates to the image mask
        below. can set CPL_DEBUG=True to see HTTP range
        requests/rasterio env/etc
        """

        with rasterio.Env():
            LOGGER.info(f"Reading raster source {raster}")
            with rasterio.open(raster) as src:
                transform = SourceWindow._adjust_affine_to_grid(src.transform, grid)
                with WarpedVRT(
                    src,
                    transform=transform,
                    resampling=Resampling.nearest,
                    height=grid.pixels,
                    width=grid.pixels,
                ) as vrt:
                    try:
                        window, shifted_affine = SourceWindow._get_window_and_affine(
                            geom, vrt, grid
                        )
                        data = vrt.read(1, masked=masked, window=window)
                        no_data_value = src.nodata
                    except MemoryError:
                        logging.exception(
                            "[ERROR][RasterAnalysis] " + traceback.format_exc()
                        )
                        raise Exception(
                            "Out of memory- input polygon or input extent too large. "
                            "Try splitting the polygon into multiple requests."
                        )

        return data, shifted_affine, no_data_value

    @staticmethod
    def _adjust_affine_to_grid(affine: Affine, grid: Grid):
        pixel_width = grid.get_pixel_width()
        return Affine(
            pixel_width, affine[1], affine[2], affine[3], -pixel_width, affine[5]
        )

    @staticmethod
    def _read_window_ignore_missing(
        raster: str, geom: BasePolygon, grid: Grid, masked: bool = False
    ) -> Tuple[np.ndarray, Affine, Numeric]:
        try:
            data = SourceWindow._read_window(raster, geom, grid, masked=masked)
        except rasterio.errors.RasterioIOError as e:
            logging.warning("RasterIO error reading " + raster + ":\n" + str(e))
            data = (
                np.zeros(
                    (grid.get_tile_width(), grid.get_tile_width()), dtype=np.uint8
                ),
                Affine(0, 0, 0, 0, 0, 0),
                0,
            )

        return data

    @staticmethod
    def _get_window_and_affine(
        geom: BasePolygon, raster_src: DatasetReader, grid: Grid
    ) -> Tuple[wd.Window, Affine]:
        """Get a rasterio window block from the bounding box of a vector
        feature and calculates the affine transformation needed to map the
        coordinates of the geometry onto a resulting array defined by the shape
        of the window.

        Args:
            geom (Shapely geometry): A geometry in the spatial reference system
                of the raster to be read.

            raster_src (rasterio file-like object): A rasterio raster source which
                will have the window operation performed and contains the base
                affine transformation.

        Returns:
            A pair of tuples which define a rectangular range that can be provided
            to rasterio for a windowed read
            See: https://mapbox.github.io/rasterio/windowed-rw.html#windowrw

            An Affine object used to transform geometry coordinates to cell values
        """

        # Create a window range from the bounds
        window: wd.Window = (
            raster_src.window(*geom.bounds)
            .round_lengths(pixel_precision=5)
            .round_offsets(pixel_precision=5)
        )

        # Create a transform relative to this window
        affine = rasterio.windows.transform(window, raster_src.transform)

        # # change pixel width to grid size
        # pixel_width = grid.get_pixel_width()
        # affine = Affine(pixel_width, affine[1], affine[2], affine[3], -pixel_width, affine[5])

        return window, affine


class DerivedWindow(SourceWindow):
    def __init__(self, layer: DerivedLayer, source_window: SourceWindow, area: float):
        A = source_window.data
        self.data = eval(layer.derivation_expression)
