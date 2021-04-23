import math

from shapely.geometry import Point, Polygon

from enum import Enum
from pydantic import BaseModel


class TileScheme(str, Enum):
    nw = "nw"
    nwse = "nwse"


class GridName(str, Enum):
    one_by_four_thousand = "1/4000"
    three_by_thirty_three_thousand_six_hundred = "3/33600"
    three_by_fifty_thousand = "3/50000"
    eight_by_thirty_two_thousand = "8/32000"
    ten_by_forty_thousand = "10/40000"
    ten_by_one_hundred_thousand = "10/100000"
    ninety_by_nine_thousand_nine_hundred_eighty_four = "90/9984"
    ninety_by_twenty_seven_thousand_eight = "90/27008"


class Grid(BaseModel):
    degrees: int
    pixels: int
    tile_degrees: float

    @staticmethod
    def get_grid(name: GridName):
        degrees, pixels = name.split("/")
        tile_degrees = degrees * (5000 / pixels)
        return Grid(degrees, pixels, tile_degrees)

    def get_pixel_width(self) -> float:
        return self.degrees / self.pixels

    def get_tile_width(self) -> int:
        return round((self.tile_degrees / self.degrees) * self.pixels)

    def get_tile_id(self, geometry: Polygon, tile_scheme: TileScheme) -> str:
        """
        Get name of tile in data lake centroid of geometry falls in

        :param: Shapely Polygon
        :return: tile id
        """
        centroid = geometry.centroid
        if tile_scheme == TileScheme.nw:
            return self._get_nw_tile_id(centroid, self.degrees)
        elif tile_scheme == TileScheme.nwse:
            return self._get_nwse_tile_id(centroid, self.degrees)

    @staticmethod
    def _get_nw_tile_id(point: Point, grid_size) -> str:
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

    @staticmethod
    def _get_nwse_tile_id(tile) -> str:
        left, bottom, right, top = tile.bounds

        left = Grid._lower_bound(left)
        bottom = Grid._lower_bound(bottom)
        right = Grid._upper_bound(right)
        top = Grid._upper_bound(top)

        west = Grid._get_longitude(left)
        south = Grid._get_latitude(bottom)
        east = Grid._get_longitude(right)
        north = Grid._get_latitude(top)

        return f"{west}_{south}_{east}_{north}"

    @staticmethod
    def _get_longitude(x: int) -> str:
        if x >= 0:
            return str(x).zfill(3) + "E"
        else:
            return str(-x).zfill(3) + "W"

    @staticmethod
    def _get_latitude(y: int) -> str:
        if y >= 0:
            return str(y).zfill(2) + "N"
        else:
            return str(-y).zfill(2) + "S"

    @staticmethod
    def _lower_bound(y: int) -> int:
        return int(math.floor(y / 10) * 10)

    @staticmethod
    def _upper_bound(y: int) -> int:
        if y == Grid._lower_bound(y):
            return int(y)
        else:
            return int((math.floor(y / 10) * 10) + 10)


# def get_raster_uri(layer: Layer, tile: Polygon) -> str:
#     """
#     Maps layer name input to a raster URI in the data lake
#     :param layer: Either of format <layer name>__<unit> or <unit>__<layer>
#     :return: A GDAL (vsis3) URI to the corresponding VRT for the layer in the data lake
#     """
#
#     if "umd_glad_landsat_alerts" in layer.layer:
#         return _get_glad_raster_uri(tile)
#
#     parts = layer.layer.split("__")
#
#     if len(parts) != 2:
#         raise ValueError(
#             f"Layer name `{layer.layer}` is invalid data lake layer, should consist of layer name and unit separated by `__`"
#         )
#
#     if parts[0] == "is":
#         type, name = parts
#     else:
#         name, type = parts
#
#     tile_id = get_tile_id(tile)
#     version = layer.version
#     return f"/vsis3/gfw-data-lake/{name}/{version}/raster/epsg-4326/{layer.grid.degrees}/{layer.grid.pixels}/{type}/gdal-geotiff/{tile_id}.tif"
#
#
# def _get_glad_raster_uri(tile: Polygon) -> str:
#     # return hardcoded URL
#     tile_id = _get_glad_tile_id(tile)
#     return (
#         f"s3://gfw2-data/forest_change/umd_landsat_alerts/prod/analysis/{tile_id}.tif"
#     )
