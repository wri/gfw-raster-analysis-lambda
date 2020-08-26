import logging
import traceback
from typing import Tuple

import numpy as np
import rasterio
from rasterio import features, DatasetReader
from rasterio.transform import Affine
from aws_xray_sdk.core import xray_recorder
from rasterio.windows import Window

from raster_analysis.globals import LOGGER, BasePolygon, Numeric


@xray_recorder.capture("Read Window")
def read_window(
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
                window, shifted_affine = get_window_and_affine(geom, src)
                data = src.read(1, masked=masked, window=window)
                no_data_value = src.nodata
            except MemoryError:
                logging.error("[ERROR][RasterAnalysis] " + traceback.format_exc())
                raise Exception(
                    "Out of memory- input polygon or input extent too large. "
                    "Try splitting the polygon into multiple requests."
                )

    return data, shifted_affine, no_data_value


def read_window_ignore_missing(
    raster: str, geom: BasePolygon, masked: bool = False
) -> Tuple[np.ndarray, Affine, Numeric]:
    try:
        data = read_window(raster, geom, masked=masked)
    except rasterio.errors.RasterioIOError as e:
        logging.warning("RasterIO error reading " + raster + ":\n" + str(e))
        data = np.array([]), None, None

    return data


@xray_recorder.capture("Mask Geometry")
def mask_geom_on_raster(
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


def get_window_and_affine(
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
