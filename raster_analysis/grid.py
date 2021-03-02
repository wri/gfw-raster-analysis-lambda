import math

from shapely.geometry import Point, Polygon

from raster_analysis.globals import DATA_LAKE_LAYER_MANAGER, GRID_SIZE, GRID_COLS
from raster_analysis.query import LayerInfo


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


def get_raster_uri(layer: LayerInfo, tile: Polygon) -> str:
    """
    Maps layer name input to a raster URI in the data lake
    :param layer: Either of format <layer name>__<unit> or <unit>__<layer>
    :return: A GDAL (vsis3) URI to the corresponding VRT for the layer in the data lake
    """

    tile_id = get_tile_id(tile)
    version = DATA_LAKE_LAYER_MANAGER.layers[layer].version
    return f"/vsis3/gfw-data-lake/{layer.name}/{version}/raster/epsg-4326/{GRID_SIZE}/{GRID_COLS}/{layer.type}/gdal-geotiff/{tile_id}.tif"
