import logging
import traceback

import numpy as np
import rasterio
from rasterio import features
from aws_xray_sdk.core import xray_recorder

from raster_analysis.grid import get_raster_url
from raster_analysis.geodesy import get_area
from raster_analysis.exceptions import RasterReadException

from collections import namedtuple

import threading
import queue
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
RasterWindow = namedtuple("RasterWindow", "data shifted_affine no_data")


@xray_recorder.capture("Read All Windows")
def read_windows_parallel(
    raster_ids, geom, analysis_raster_id, masked=False, get_area_raster=False
):
    read_window_threads = []
    result_queue = queue.Queue()
    error_queue = queue.Queue()

    for raster_id in raster_ids:
        read_window_thread = threading.Thread(
            target=read_window_parallel_work,
            args=(raster_id, geom, masked, result_queue, error_queue),
        )
        read_window_thread.start()
        read_window_threads.append(read_window_thread)

    if get_area_raster:
        get_area_thread = threading.Thread(
            target=create_area_raster_work,
            args=(geom, analysis_raster_id, result_queue, error_queue),
        )
        get_area_thread.start()
        read_window_threads.append(get_area_thread)

    for read_window_thread in read_window_threads:
        read_window_thread.join()

    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())

    if errors:
        raise RasterReadException("\n".join([str(e) for e in errors]))

    result_dict = {}
    while not result_queue.empty():
        result = result_queue.get()
        result_dict[result[0]] = RasterWindow(
            data=result[1], shifted_affine=result[2], no_data=result[3]
        )

    return result_dict


@xray_recorder.capture("Calculate Pixel Areas")
def create_area_raster_work(geom, dummy_raster, result_queue, error_queue):
    try:
        with rasterio.Env():
            with rasterio.open(get_raster_url(dummy_raster)) as src:
                window, affine = get_window_and_affine(geom, src)

                height = int(math.floor(window.height))
                width = int(math.floor(window.width))

                base_matrix = np.ones((height, width), dtype=np.uint8)
                y_indices = np.indices((height, 1))[0]
                lat_coords = _get_lat_coords(y_indices, affine)
                pixel_areas = get_area(lat_coords) / 10000
                area_matrix = base_matrix * pixel_areas

                result_queue.put(("area", area_matrix, affine, 0))
    except rasterio.errors.RasterioIOError as e:
        logging.warning(e)
        result_queue.put(("area", np.array([]), None, None))
    except Exception as e:
        error_queue.put(e)


def _get_lat_coords(y_indices, affine):
    return y_indices * -0.00025 + affine[5] + (-0.00025 / 2)


def read_window_parallel_work(raster_id, geom, masked, result_queue, error_queue):
    try:
        raster_url = get_raster_url(raster_id)

        data, shifted_affine, no_data_value = read_window_ignore_missing(
            raster_url, geom, masked=masked
        )

        result_queue.put((raster_id, data, shifted_affine, no_data_value))
    except Exception as e:
        error_queue.put(e)


@xray_recorder.capture("Read Window")
def read_window(raster, geom, masked=False):
    """
    Read a chunk of the raster that contains the bounding box of the
    input geometry.  This has memory implications if that rectangle
    is large. The affine transformation maps geom coordinates to the
    image mask below.
    can set CPL_DEBUG=True to see HTTP range requests/rasterio env/etc
    """

    with rasterio.Env():
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


def read_window_ignore_missing(raster, geom, masked=False):
    try:
        data = read_window(raster, geom, masked=masked)
    except rasterio.errors.RasterioIOError as e:
        logging.warning("RasterIO error reading " + raster + ":\n" + str(e))
        data = np.array([]), None, None

    return data


@xray_recorder.capture("Mask Geometry")
def mask_geom_on_raster(raster_data, shifted_affine, geom):
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

    if raster_data.any():

        # Create a numpy array to mask cells which don't intersect with the
        # polygon. Cells that intersect will have value of 1 (unmasked), the
        # rest are filled with 0s (masked)
        geom_mask = features.geometry_mask(
            [geom], out_shape=raster_data.shape, transform=shifted_affine, invert=True
        )

        # Mask the data array, with modifications applied, by the query polygon
        return geom_mask

    else:
        return np.array([]), np.array([])


def get_window_and_affine(geom, raster_src):
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
    window = raster_src.window(*geom.bounds)
    # Create a transform relative to this window
    affine = rasterio.windows.transform(window, raster_src.transform)

    return window, affine
