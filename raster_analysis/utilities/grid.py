import math

BASE_URL = "/vsis3/gfw-files/2018_update/{raster_id}/{tile_id}.tif"


def get_grid_id(point, grid_size=10):
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


def get_tile_id(geometry):
    centroid = geometry.centroid
    return get_grid_id(centroid)


def get_raster_url(raster_id, tile_id):
    return BASE_URL.format(raster_id=raster_id, tile_id=tile_id)