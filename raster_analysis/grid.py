import math

from shapely.geometry import Point, Polygon

from raster_analysis.layer import Layer
from raster_analysis.globals import GRID_SIZE, GRID_COLS


def _get_tile_id(point: Point, grid_size=10) -> str:
    """
    Get name of tile in data lake

    :param point: Shapely point
    :param grid_size: Tile size of grid to check against
    :return:
    """
    col = int(math.floor(point.x / grid_size)) * grid_size
    if col >= 0:

        long = str(col).zfill(3) + "E"
    else:
        long = str(-col).zfill(3) + "W"

    row = int(math.ceil(point.y / grid_size)) * grid_size

    if row >= 0:
        lat = str(row).zfill(2) + "N"
    else:
        lat = str(-row).zfill(2) + "S"

    return f"{lat}_{long}"


def get_tile_id(geometry: Polygon) -> str:
    """
    Get name of tile in data lake centroid of geometry falls in

    :param: Shapely Polygon
    :return: tile id
    """
    centroid = geometry.centroid
    return _get_tile_id(centroid)


def get_raster_uri(layer: Layer, tile: Polygon) -> str:
    """
    Maps layer name input to a raster URI in the data lake
    :param layer: Either of format <layer name>__<unit> or <unit>__<layer>
    :return: A GDAL (vsis3) URI to the corresponding VRT for the layer in the data lake
    """

    if "umd_glad_alerts" in layer.layer:
        return _get_glad_raster_uri(tile)

    parts = layer.layer.split("__")

    if len(parts) != 2:
        raise ValueError(
            f"Layer name `{layer.layer}` is invalid data lake layer, should consist of layer name and unit separated by `__`"
        )

    if parts[0] == "is":
        type, name = parts
    else:
        name, type = parts

    tile_id = get_tile_id(tile)
    version = layer.version
    return f"/vsis3/gfw-data-lake/{name}/{version}/raster/epsg-4326/{GRID_SIZE}/{GRID_COLS}/{type}/gdal-geotiff/{tile_id}.tif"


def _get_glad_raster_uri(tile: Polygon) -> str:
    # return hardcoded URL
    tile_id = _get_glad_tile_id(tile)
    return f"s3://gfw2-data/forest_change/umd_landsat_alerts/prod/analysis/{tile_id}.tif"


def _get_glad_tile_id(tile) -> str:
    left, bottom, right, top = tile.bounds

    left = _lower_bound(left)
    bottom = _lower_bound(bottom)
    right = _upper_bound(right)
    top = _upper_bound(top)

    west = _get_longitude(left)
    south = _get_latitude(bottom)
    east = _get_longitude(right)
    north = _get_latitude(top)

    return f"{west}_{south}_{east}_{north}"


def _get_longitude(x: int) -> str:
    if x >= 0:
        return str(x).zfill(3) + "E"
    else:
        return str(-x).zfill(3) + "W"


def _get_latitude(y: int) -> str:
    if y >= 0:
        return str(y).zfill(2) + "N"
    else:
        return str(-y).zfill(2) + "S"


def _lower_bound(y: int) -> int:
    return int(math.floor(y / 10) * 10)


def _upper_bound(y: int) -> int:
    if y == _lower_bound(y):
        return int(y)
    else:
        return int((math.floor(y / 10) * 10) + 10)