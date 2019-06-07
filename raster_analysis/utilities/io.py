import os
import json

import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import Polygon

# from raster_analysis.utilities.errors import Error
import math
import logging


def read_window(raster, geom, masked=False):
    # Read a chunk of the raster that contains the bounding box of the
    # input geometry.  This has memory implications if that rectangle
    # is large. The affine transformation maps geom coordinates to the
    # image mask below.
    # can set CPL_DEBUG=True to see HTTP range requests/rasterio env/etc

    with rasterio.Env():
        with rasterio.open(raster) as src:
            try:
                window, shifted_affine = get_window_and_affine(geom, src)
                data = src.read(1, masked=masked, window=window)
                no_data_value = src.nodata
            except MemoryError:
                raise Exception(
                    "Out of memory- input polygon or input extent too large. "
                    "Try splitting the polygon into multiple requests."
                )
    return data, shifted_affine, no_data_value


def read_window_ignore_missing(raster, geom, masked=False):
    try:
        data = read_window(raster, geom, masked=masked)
    except rasterio.errors.RasterioIOError as e:
        logging.warning(e)
        data = np.array([]), None, None

    return data


def check_extent(user_poly, raster):
    raster_ext = os.path.splitext(raster)[1]
    geojson_src = raster.replace(raster_ext, ".geojson")

    with open(geojson_src) as src:
        d = json.load(src)

    # get index geom
    poly_intersects = False
    poly_list = [Polygon(x["geometry"]["coordinates"][0]) for x in d["features"]]

    # check if polygons intersect
    for poly in poly_list:
        if user_poly.intersects(poly):
            poly_intersects = True
            break

    return poly_intersects


def mask_geom_on_raster(geom, raster_path):
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

    data, shifted_affine, no_data_value = read_window_ignore_missing(
        raster_path, geom, masked=True
    )

    if data.any():

        # Create a numpy array to mask cells which don't intersect with the
        # polygon. Cells that intersect will have value of 1 (unmasked), the
        # rest are filled with 0s (masked)
        geom_mask = features.geometry_mask(
            [geom], out_shape=data.shape, transform=shifted_affine, invert=True

        )

        # Include any NODATA mask
        full_mask = geom_mask | data.mask

        # Mask the data array, with modifications applied, by the query polygon
        return np.ma.array(data=data, mask=full_mask), shifted_affine, no_data_value

    else:
        return np.array([]), None, None


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
    ul = raster_src.index(*geom.bounds[0:2])
    lr = raster_src.index(*geom.bounds[2:4])
    window = ((lr[0], ul[0] + 1), (ul[1], lr[1] + 1))
    # window = ((lr[0], ul[0]), (ul[1], lr[1])) # TODO: figure out why we have to extent the bounds

    # Create a transform relative to this window
    affine = rasterio.windows.transform(window, raster_src.transform)

    return window, affine


def array_to_xyz_rows(arr, shifted_affine):
    i, j = np.where(arr.mask == False)
    masked_x = j * 0.00025 + shifted_affine.xoff + 0.000125
    masked_y = i * -0.00025 + shifted_affine.yoff - 0.000125

    for x, y, z in zip(masked_x, masked_y, arr.compressed()):
        yield (x, y, z)
