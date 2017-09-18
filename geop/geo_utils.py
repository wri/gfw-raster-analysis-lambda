from __future__ import division

from functools import partial
from rasterio import features
from shapely.ops import transform, cascaded_union
from shapely.geometry import shape, mapping
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.geo import box

import numpy as np
import pyproj
import rasterio


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


def subdivide_polygon(geom, factor):
    """
    Divide a geometry such that no piece is greater than the size of
    `factor`, in units of the coordinate system.

    Args:
        geom: GeoJson-like polygon to subdivide
        factor: The number of SRS units to divide the geom bounds by,
            to provide subgeometries who's extent does not exceed that size.
    Returns:
        List of GeoJson-like polygons that `geom` is composed of
    """
    bounds = np.asarray(geom.bounds)
    xmin, ymin, xmax, ymax = np.floor_divide(bounds, factor).astype(int)

    children = []
    for i in range(xmin, xmax + 1):
        for j in range(ymin, ymax + 1):
            sub_poly = box(i * factor, j * factor,
                           (i + 1) * factor, (j + 1) * factor)
            overlap_poly = geom.intersection(sub_poly)

            if not overlap_poly.is_empty:
                children.append(overlap_poly)

    return children


def reproject(geom, from_srs, to_srs):

    projection = partial(
        pyproj.transform,
        pyproj.Proj(init=from_srs),
        pyproj.Proj(init=to_srs),
    )

    return transform(projection, geom)


def color_table_to_palette(src):
    """
    Convert an RGBA raster color table to a PIL appropriate palette.

    Args:
        src (RasterioReader): An open raster reader that contains a colortable
            of integer values mapped to RGB (or RGBA, though A will be ignored)

    Returns:
        ndarray of RGB sequences whose root index maps to a cell value.
        ie (0,0,0,255,255,255) maps to 0: RGB(0,0,0), 1: (255, 255, 255)
    """
    color_len = 3
    bit_len = 255
    palette = np.zeros(bit_len * color_len + color_len, dtype=np.uint8)
    try:
        for cell_val, rgb in src.colormap(1).iteritems():
            for idx in range(color_len):
                palette_index = cell_val * color_len + idx
                palette[palette_index] = rgb[idx]

        return palette
    except ValueError:
        return None


def tile_read(geom, raster_path):
    """
    Decimated read against raster_path to fit into a 256x256 ndarry tile.
    Resampling method is NEAREST NEIGHBOR and is not configurable.  This allows
    for very high zoom level tiles to be rendered at reasonable performance
    without chance of memory errors.

    Args:
        geom (Shapely Geometry): Polygon representing the geographic envelope
            of the tile to be rendered

        raster_path (string): Path to raster to read at geom.  Must be in
            EPSG:3857.  If raster contains and integer ColorTable, a PIL
            palette will be returned with those values

    Returns:
        ndarray of decimated read of source raster in a EPSG:3857 transformed
            grid

        palette (ndarray uint8) of RGB colors defined in raster ColorTable
    """
    tile_size = 256
    with rasterio.open(raster_path) as src:
        window, _ = get_window_and_affine(geom, src)
        tile = src.read(1, window=window, out_shape=(1, tile_size, tile_size))

        palette = color_table_to_palette(src)
        return tile, palette


def tile_to_bbox(zoom, x, y):
    """
    Transform a TMS/Slippy Map style tile protocol (z/x/y) to a web mercator
    bounding box.

    Ref:
        https://github.com/IzAndCuddles/gdal2tiles/blob/structure/gdal2tiles.py#L120-L146  # noqa

    Args:
        zoom (int): Zoom level for the tile
        x (int): x coordinate of tile origin
        y (int): y coordinate of tile origin

    Returns:
        A Shapely geometry in EPSG:3857 defining the bounding box of the requested
        tile
    """
    mapSize = 20037508.34789244 * 2
    origin_x = -20037508.34789244
    origin_y = 20037508.34789244
    size = mapSize / 2**zoom

    min_x = origin_x + x*size
    min_y = origin_y - (y+1)*size
    max_x = origin_x + (x+1)*size
    max_y = origin_y - y*size

    return box(min_x, min_y, max_x, max_y, ccw=False)


def interpolate_points(line):
    """
    Break a line into linear points
    """
    return [line.interpolate(n/150, normalized=True).coords
            for n in range(0, 150, 1)]


def as_json(geoms, from_srs='epsg:5070', to_srs='epsg:4326'):
    """
    Return a list of shapely objects as a reprojected GeoJSON
    FeatureCollection

    Args:
        geoms: list of shapely geometries
        from_srs: EPSG Code of provided geometries (5070)
        to_srs: EPSG Code of desired output geometries (4326)
    """
    features = [reproject(shape(geom), to_srs, from_srs)
                for geom in geoms]

    return mapping(cascaded_union(features))
