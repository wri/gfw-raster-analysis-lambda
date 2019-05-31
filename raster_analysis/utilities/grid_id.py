import math


def get_gridid(point, grid_size=10):
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
