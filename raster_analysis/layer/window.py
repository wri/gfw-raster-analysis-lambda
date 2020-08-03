import numpy as np
from rasterio import Affine
from shapely.geometry import Polygon
from numpy import ndarray
from math import floor

from raster_analysis.grid import get_raster_uri
from raster_analysis.io import read_window

from datetime import date, timedelta
from typing import Union, Tuple

Numeric = Union[int, float]
WINDOW_SIZE = 4000


class Window:
    def __init__(self, layer: str, tile: Polygon):
        self.layer: LayerInfo = LayerInfo(layer)
        self.tile: Polygon = tile

        self.data: ndarray = None
        self.shifted_affine: Affine = None
        self.no_data_value: Numeric = 0

        self.data, self.shifted_affine, self.no_data_value = self.read(tile)
        self._result: ndarray = None

    def read(self, tile: Polygon) -> Tuple[np.ndarray, Affine, Numeric]:
        data, shifted_affine, no_data_value = read_window(self.get_raster_uri(), tile)

        if data.size == 0:
            data = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.uint8)

        return data, shifted_affine, no_data_value

    def sum(self, mean_area, mask, linear_index=None, index_counts=None):
        data = np.extract(mask, self.data)

        if linear_index is None and index_counts is None:
            return data.sum()

        return np.bincount(linear_index, weights=data, minlength=index_counts.size)

    def clear(self):
        self.data = []

    @property
    def result(self):
        return self.layer.name, self._result.tolist()

    @result.setter
    def result(self, value):
        self._result = value

    def get_raster_uri(self):
        return get_raster_uri(self.layer.layer, self.layer.data_type, self.tile)


def get_window(layer: str, tile: Polygon) -> Window:
    if layer == "area__ha":
        return AreaWindow(layer, tile)
    elif layer == "alert__count":
        return CountWindow(layer, tile)
    elif layer.startswith("umd_glad_alerts"):
        return GladAlertsWindow(layer, tile)
    elif layer.endswith("__year"):
        return YearWindow(layer, tile)
    elif layer.startswith("umd_tree_cover_density"):
        return TcdWindow(layer, tile)
    elif "__ha-1" in layer:
        return AreaDensityWindow(layer, tile)
    elif "whrc_aboveground_co2_emissions" in layer:
        return CarbonEmissionsWindow(layer, tile)
    else:
        return Window(layer, tile)


class YearWindow(Window):
    """
    Class representing year layers, which is always encoded as (year - 2000)
    """

    @property
    def result(self):
        return super().value

    @result.setter
    def result(self, value):
        self._result = value + 2000


class TcdWindow(Window):
    COMPRESSED_THRESHOLD_MAP = {
        "10": 1,
        "15": 2,
        "20": 3,
        "25": 4,
        "30": 5,
        "50": 6,
        "75": 7,
    }

    def __init__(self, layer: str, tile: Polygon):
        name, self.threshold = layer.split("__")

        super().__init__(f"{name}__threshold", tile)

        self.data = self.data >= self.COMPRESSED_THRESHOLD_MAP[self.threshold]


class GladAlertsWindow(Window):
    def __init__(self, layer: str, tile: Polygon):
        name, self.agg = layer.split("__")
        self.confirmed = "confirmed" in name

        super().__init__(layer, tile)

        # if only confirmed, remove any value beneath 30000 (which is unconfirmed)
        if self.confirmed:
            self.data[self.data < 30000] = 0

        # remove conf and set to ordinal date since 2015
        self.data %= 10000

    def get_raster_uri(self):
        # return hardcoded URL
        tile_id = self.get_tile_id()
        return f"s3://gfw2-data/forest_change/umd_landsat_alerts/prod/analysis/{tile_id}.tif"

    def get_tile_id(self):
        left, bottom, right, top = self.tile.bounds

        left = self.lower_bound(left)
        bottom = self.lower_bound(bottom)
        right = self.upper_bound(right)
        top = self.upper_bound(top)

        west = self.get_longitude(left)
        south = self.get_latitude(bottom)
        east = self.get_longitude(right)
        north = self.get_latitude(top)

        return f"{west}_{south}_{east}_{north}"

    @staticmethod
    def get_longitude(x):
        if x >= 0:
            return str(x).zfill(3) + "E"
        else:
            return str(-x).zfill(3) + "W"

    @staticmethod
    def get_latitude(y):
        if y >= 0:
            return str(y).zfill(2) + "N"
        else:
            return str(-y).zfill(2) + "S"

    @staticmethod
    def lower_bound(y):
        return int(floor(y / 10) * 10)

    @staticmethod
    def upper_bound(y):
        if y == GladAlertsWindow.lower_bound(y):
            return int(y)
        else:
            return int((floor(y / 10) * 10) + 10)

    @property
    def result(self):
        # return ("iso_week", self._result)
        return super().value

    @result.setter
    def result(self, value):
        # value is already ordinal date sine 2015, so just need to add 2015 ordinal date to get iso date
        value += date(2015, 1, 1).toordinal()
        value = [date.fromordinal(ordinal) for ordinal in value]

        if self.agg == "isoweek":
            # change to ordinal date of beginning of iso week
            value = [
                (d - timedelta(days=d.isoweekday() - 1)).toordinal() for d in value
            ]

        self._result = np.array(value)

        # results = {}
        # days_since_2015 =results[col][i] - GLAD_CONFIRMED_CONST
        # raw_date = date(2015, 1, 1) + timedelta(days=days_since_2015)
        # row["alert__date"] = raw_date.strftime("%Y-%m-%d")
        # ORDINAL_2015 = 16436
        # date_values = (value + ORDINAL_2015).astype('datetime64[D]')

    def _decode_date(self, date):
        return date


class CountWindow(Window):
    def __init__(self, layer: str, tile: Polygon):
        self.layer: LayerInfo = LayerInfo(layer)
        self.tile = tile
        self.data = None

    def sum(self, mean_area, mask, linear_index=None, index_counts=None):
        if linear_index is None and index_counts is None:
            return mask.sum()

        return index_counts


class AreaWindow(CountWindow):
    def sum(self, mean_area, mask, linear_index=None, index_counts=None):
        if linear_index is None and index_counts is None:
            return mask.sum() * mean_area

        return index_counts * mean_area


class AreaDensityWindow(Window):
    def sum(self, mean_area, mask, linear_index=None, index_counts=None):
        self.data = self.data.astype(np.float32) * mean_area
        return super().sum(mean_area, mask, linear_index, index_counts)


class CarbonEmissionsWindow(Window):
    CO2_FACTOR = 0.5 * 44 / 12  # used to calculate emissions from biomass layer
    BIOMASS_LAYER = "whrc_aboveground_biomass_stock_2000"
    BIOMASS_TYPE = "Mg_ha-1"

    def sum(self, mean_area, mask, linear_index=None, index_counts=None):
        self.data = self.data.astype(np.float32) * self.CO2_FACTOR * mean_area
        return super().sum(mean_area, mask, linear_index, index_counts)

    def get_raster_uri(self):
        return get_raster_uri(self.BIOMASS_LAYER, self.BIOMASS_TYPE, self.tile)


class LayerInfo:
    def __init__(self, name: str):
        self.name = name
        parts = name.split("__")

        if len(parts) != 2:
            raise ValueError(
                f"Layer name `{name}` is invalid, should consist of layer name and unit separated by `__`"
            )

        if parts[0] == "is":
            self.data_type, self.layer = parts
        else:
            self.layer, self.data_type = parts
