import os
import json

import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import Polygon, MultiPolygon

from utilities.errors import Error
import math


def check_extent(user_poly, raster):

    raster_ext = os.path.splitext(raster)[1]
    geojson_src = raster.replace(raster_ext, '.geojson')

    with open(geojson_src) as src:
        d = json.load(src)

    # get index geom
    poly_intersects = False
    poly_list = [Polygon(x['geometry']['coordinates'][0]) for x in d['features']]

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

    if check_extent(geom, raster_path):

        # Read a chunk of the raster that contains the bounding box of the
        # input geometry.  This has memory implications if that rectangle
        # is large. The affine transformation maps geom coordinates to the
        # image mask below.
        # can set CPL_DEBUG=True to see HTTP range requests/rasterio env/etc
        with rasterio.Env():
            with rasterio.open(raster_path) as src:

                window, shifted_affine = get_window_and_affine(geom, src)

                try:
                    data = src.read(1, masked=True, window=window)
                except MemoryError:
                    raise Error('Out of memory- input polygon or input extent too large. ' 
                                'Try splitting the polygon into multiple requests.')

                no_data = src.nodata

            # Create a numpy array to mask cells which don't intersect with the
            # polygon. Cells that intersect will have value of 0 (unmasked), the
            # rest are filled with 1s (masked)
            geom_mask = features.geometry_mask(
                [geom],
                out_shape=data.shape,
                transform=shifted_affine
            )

            # Include any NODATA mask
            full_mask = geom_mask | data.mask

            # Mask the data array, with modifications applied, by the query polygon
            return np.ma.array(data=data, mask=full_mask), no_data

    else:
        return np.array([]), None


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
    window = ((lr[0], ul[0]+1), (ul[1], lr[1]+1))

    # Create a transform relative to this window
    affine = rasterio.windows.transform(window, raster_src.transform)

    return window, affine


def array_to_xyz_rows(arr, shifted_affine):

    i, j = np.where(arr.mask == False)
    masked_x = j * .00025 + shifted_affine.xoff + 0.000125
    masked_y = i * -.00025 + shifted_affine.yoff - 0.000125

    for x, y, z in zip(masked_x, masked_y, arr.compressed()):
        yield (x, y, z)


def get_area(lat):
    """
    Calculate geodesic area for Hansen data, assuming a fix pixel size of 0.00025 * 0.00025 degree
    using WGS 1984 as spatial reference.
    Pixel size various with latitude, which is it the only input parameter.
    """
    a = 6378137.0  # Semi major axis of WGS 1984 ellipsoid
    b = 6356752.314245179  # Semi minor axis of WGS 1984 ellipsoid

    d_lat = 0.00025  # pixel hight
    d_lon = 0.00025  # pixel width

    pi = math.pi

    q = d_lon / 360
    e = math.sqrt(1 - (b / a) ** 2)

    area = abs(
        (pi * b ** 2 * (
                2 * np.arctanh(e * np.sin(np.radians(lat + d_lat))) /
                (2 * e) +
                np.sin(np.radians(lat + d_lat)) /
                ((1 + e * np.sin(np.radians(lat + d_lat))) * (1 - e * np.sin(np.radians(lat + d_lat)))))) -
        (pi * b ** 2 * (
                2 * np.arctanh(e * np.sin(np.radians(lat))) / (2 * e) +
                np.sin(np.radians(lat)) / ((1 + e * np.sin(np.radians(lat))) * (1 - e * np.sin(np.radians(lat))))))) *q

    return area