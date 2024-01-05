import math
from enum import Enum

from shapely.geometry import Point, Polygon

from raster_analysis.globals import BasePolygon


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


class Grid:
    def __init__(self, degrees: int, pixels: int, tile_degrees: float):
        self.degrees = degrees
        self.pixels = pixels
        self.tile_degrees = tile_degrees

    @classmethod
    def get_grid(cls, name: GridName):
        degrees_str, pixels_str = name.split("/")
        degrees = int(degrees_str)
        pixels = int(pixels_str)
        tile_degrees = degrees * (2000 / pixels)
        return cls(degrees, pixels, tile_degrees)

    def get_pixel_width(self) -> float:
        return self.degrees / self.pixels

    def get_tile_width(self) -> int:
        return round((self.tile_degrees / self.degrees) * self.pixels)

    def get_tile_id(self, geometry: Polygon, tile_scheme: TileScheme) -> str:
        """Get name of tile in data lake centroid of geometry falls in.

        :param: Shapely Polygon
        :return: tile id
        """
        if tile_scheme == TileScheme.nw:
            centroid = geometry.centroid
            return self._get_nw_tile_id(centroid, self.degrees)
        elif tile_scheme == TileScheme.nwse:
            return self._get_nwse_tile_id(geometry, self.degrees)
        else:
            raise NotImplementedError(f"Tile scheme {tile_scheme} not implemented.")

    def _get_nw_tile_id(self, point: Point, grid_size) -> str:
        """Get name of tile in data lake.

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

    def _get_nwse_tile_id(self, tile: BasePolygon, grid_size: int) -> str:
        left, bottom, right, top = tile.bounds

        left = self._lower_bound(left, grid_size)
        bottom = self._lower_bound(bottom, grid_size)
        right = self._upper_bound(right, grid_size)
        top = self._upper_bound(top, grid_size)

        west = self._get_longitude(left)
        south = self._get_latitude(bottom)
        east = self._get_longitude(right)
        north = self._get_latitude(top)

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
    def _lower_bound(y: int, grid_size: int) -> int:
        return int(math.floor(y / grid_size) * grid_size)

    @staticmethod
    def _upper_bound(y: int, grid_size: int) -> int:
        if y == Grid._lower_bound(y, grid_size):
            return int(y)
        else:
            return int((math.floor(y / grid_size) * grid_size) + grid_size)
