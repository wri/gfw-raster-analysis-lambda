import logging
import traceback
from math import floor
from typing import Tuple

import numpy as np
import rasterio
from numpy import ndarray
from rasterio import features, DatasetReader
from rasterio.transform import Affine
from aws_xray_sdk.core import xray_recorder
from rasterio.windows import Window
from shapely.geometry import Polygon

from raster_analysis.globals import LOGGER, BasePolygon, Numeric, WINDOW_SIZE
from raster_analysis.grid import get_raster_uri
from raster_analysis.query import LayerInfo


class Window:
    def __init__(self, layer: LayerInfo, tile: Polygon):
        self.layer = layer
        self.tile = tile

        data, shifted_affine, no_data_value = self.read()

        if data.size == 0:
            self.empty = True
            data = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.uint8)
        else:
            self.empty = False

        self.data: ndarray = data
        self.shifted_affine: Affine = shifted_affine
        self.no_data_value: Numeric = no_data_value

    def read(self) -> Tuple[np.ndarray, Affine, Numeric]:
        raster_uri = get_raster_uri(self.layer, self.tile)
        data, shifted_affine, no_data_value = Window._read_window_ignore_missing(
            raster_uri, self.tile
        )

        return data, shifted_affine, no_data_value

    def clear(self) -> None:
        """
        Clear internal data array to save memory.
        """
        self.data = []


    @xray_recorder.capture("Read Window")
    @staticmethod
    def _read_window(
        raster: str, geom: BasePolygon, masked: bool = False
    ) -> Tuple[np.ndarray, Affine, Numeric]:
        """
        Read a chunk of the raster that contains the bounding box of the
        input geometry.  This has memory implications if that rectangle
        is large. The affine transformation maps geom coordinates to the
        image mask below.
        can set CPL_DEBUG=True to see HTTP range requests/rasterio env/etc
        """

        with rasterio.Env():
            LOGGER.info(f"Reading raster source {raster}")

            with rasterio.open(raster) as src:
                try:
                    window, shifted_affine = Window._get_window_and_affine(geom, src)
                    data = src.read(1, masked=masked, window=window)
                    no_data_value = src.nodata
                except MemoryError:
                    logging.error("[ERROR][RasterAnalysis] " + traceback.format_exc())
                    raise Exception(
                        "Out of memory- input polygon or input extent too large. "
                        "Try splitting the polygon into multiple requests."
                    )

        return data, shifted_affine, no_data_value


    @staticmethod
    def read_window_ignore_missing(
        raster: str, geom: BasePolygon, masked: bool = False
    ) -> Tuple[np.ndarray, Affine, Numeric]:
        try:
            data = Window._read_window(raster, geom, masked=masked)
        except rasterio.errors.RasterioIOError as e:
            logging.warning("RasterIO error reading " + raster + ":\n" + str(e))
            data = np.array([]), None, None

        return data

    @staticmethod
    def _get_window_and_affine(
            geom: BasePolygon, raster_src: DatasetReader
    ) -> Tuple[Window, Affine]:
        """
        Get a rasterio window block from the bounding box of a vector feature and
        calculates the affine transformation needed to map the coordinates of the
        geometry onto a resulting array defined by the shape of the window.

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
        window: Window = raster_src.window(*geom.bounds).round_lengths(
            pixel_precision=5
        ).round_offsets(pixel_precision=5)

        # Create a transform relative to this window
        affine = rasterio.windows.transform(window, raster_src.transform)

        return window, affine
