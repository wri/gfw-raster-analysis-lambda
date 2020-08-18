import logging
import os
import math
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


def get_grid_id(point: Point, grid_size=10) -> str:
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

    return "{}_{}".format(lat, long)


def get_tile_id(geometry: Polygon) -> str:
    """
    Get name of tile in data lake centroid of geometry falls in

    :param 5point: Shapely geometry
    :return:
    """
    centroid = geometry.centroid
    return get_grid_id(centroid)


def get_raster_uri(layer: str, data_type: str, tile: Polygon) -> str:
    """
    Maps layer name input to a raster URI in the data lake
    :param layer: Either of format <layer name>__<unit> or <unit>__<layer>
    :return: A GDAL (vsis3) URI to the corresponding VRT for the layer in the data lake
    """

    tile_id = get_tile_id(tile)
    version = LATEST_VERSIONS[layer]
    return f"/vsis3/gfw-data-lake/{layer}/{version}/raster/epsg-4326/10/40000/{data_type}/gdal-geotiff/{tile_id}.tif"


LATEST_VERSIONS = {
    "umd_tree_cover_loss": "v1.7",
    "umd_regional_primary_forest_2001": "v201901",
    "umd_tree_cover_density_2000": "v1.6",
    "umd_tree_cover_density_2010": "v1.6",
    "umd_tree_cover_gain": "v1.6",
    "whrc_aboveground_biomass_stock_2000": "v4",
    "tsc_tree_cover_loss_drivers": "v2019",
}
