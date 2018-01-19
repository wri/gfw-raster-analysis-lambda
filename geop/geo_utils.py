from functools import partial

import rasterio
from rasterio import features
import numpy as np
import pyproj
from shapely.ops import transform
import boto3
from urlparse import urlparse
import json
from shapely.geometry import Polygon, MultiPolygon


def check_extent(user_poly, raster):
    s3 = boto3.resource('s3')

    # read context of the index file
    parsed = urlparse(raster)
    bucket = s3.Bucket(parsed.netloc)

    s3obj = parsed.path.replace('data.vrt', 'index.geojson')[1:]
    d = bucket.Object(s3obj).get()['Body'].read()

    # get contents from string to dictionary
    d = json.loads(d)

    # get index geom
    poly_intersects = False
    poly_list = [Polygon(x['geometry']['coordinates'][0]) for x in d['features']]

    # check if polygons intersect
    for poly in poly_list:
        if user_poly.intersects(poly):
            poly_intersects = True
            break

    return poly_intersects
    
    
def mask_geom_on_raster(geom, raster_path, mods=None, all_touched=False):
    """"
    For a given polygon, returns a numpy masked array with the intersecting
    values of the raster at `raster_path` unmasked, all non-intersecting
    cells are masked.  This assumes that the input geometry is in the same
    SRS as the raster.  Currently only reads from a single band.

    Args:
        geom (Shapley Geometry): A polygon in the same SRS as `raster_path`
            which will define the area of the raster to mask.

        raster_path (string): A local file path to a geographic raster
            containing values to extract.

        mods (optional list): A list of modifications to make to the source
            raster, provided as json objects containing the following keys:

            geom (geojson): polygon of area where modification should be
                applied.
            newValue (int|float): value to be written over the source raster
                in areas where it intersects geom.  Modifications are applied
                in order, meaning subsequent items can overwrite earlier ones.

        all_touched (optional bool|default: True): If True, the cells value
            will be unmasked if geom interstects with it.  If False, the
            intersection must capture the centroid of the cell in order to be
            unmasked.

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
        with rasterio.open(raster_path) as src:
            window, shifted_affine = get_window_and_affine(geom, src)
            data = src.read(1, masked=True, window=window)

        # Burn new raster values in from provided vector modifications. Mods
        # are applied in order, so later polygons will overwrite previous ones
        # if they overlap
        if mods:
            # This copies over `data` in place.
            for mod in mods:
                features.rasterize(
                    [(mod['geom'], mod['newValue'])],
                    out=data,
                    transform=shifted_affine,
                    all_touched=all_touched,
                )

        # Create a numpy array to mask cells which don't intersect with the
        # polygon. Cells that intersect will have value of 0 (unmasked), the
        # rest are filled with 1s (masked)
        geom_mask = features.geometry_mask(
            [geom],
            out_shape=data.shape,
            transform=shifted_affine,
            all_touched=all_touched
        )

        # Include any NODATA mask
        full_mask = geom_mask | data.mask

        # Mask the data array, with modifications applied, by the query polygon
        return np.ma.array(data=data, mask=full_mask), shifted_affine
        
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


def get_polygon_area(geom):
    # source: https://gis.stackexchange.com/a/166421/30899

    geom_area = transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat1=geom.bounds[1],
                lat2=geom.bounds[3])),
        geom)

    # return area in ha
    return geom_area.area / 10000.
