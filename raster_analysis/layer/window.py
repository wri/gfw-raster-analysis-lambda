import numpy as np
from rasterio import Affine
from shapely.geometry import Polygon
from numpy import ndarray
from math import floor

from raster_analysis.grid import get_raster_uri
from raster_analysis.io import read_window

from datetime import date, timedelta, datetime
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
        print(f"MAX for {tile.bounds}: {data.max()}")

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


def get_window(
    layer: str, tile: Polygon, start_date: datetime, end_date: datetime
) -> Window:
    if layer == "area__ha":
        return AreaWindow(layer, tile)
    elif layer == "alert__count":
        return CountWindow(layer, tile)
    elif layer == "tsc_tree_cover_loss_drivers__type":
        return LossDriversWindow(layer, tile)
    elif layer.startswith("umd_glad_alerts"):
        return GladAlertsWindow(layer, tile, start_date, end_date)
    elif layer.endswith("__year"):
        return YearWindow(layer, tile, start_date, end_date)
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

    def __init__(
        self, layer: str, tile: Polygon, start_date: datetime, end_date: datetime
    ):
        self.start_date = start_date
        self.end_date = end_date

        super().__init__(layer, tile)

        if self.start_date:
            start_year = self.start_date.year - 2000
            self.data[self.data < start_year] = self.no_data_value

        if self.end_date:
            end_year = self.end_date.year - 2000
            self.data[self.data > end_year] = self.no_data_value

    @property
    def result(self):
        return super().result

    @result.setter
    def result(self, value):
        self._result = value + 2000


class LossDriversWindow(Window):
    """
    Class representing year layers, which is always encoded as (year - 2000)
    """

    LOSS_DRIVER_MAP = {
        2: "Commodity driven deforestation",
        3: "Shifting agriculture",
        4: "Forestry",
        5: "Wildfire",
        6: "Urbanization",
    }

    @property
    def result(self):
        return super().result

    @result.setter
    def result(self, value):
        self._result = np.array(
            [
                self.LOSS_DRIVER_MAP[val] if val in self.LOSS_DRIVER_MAP else "Unknown"
                for val in value
            ]
        )


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
    def __init__(
        self, layer: str, tile: Polygon, start_date: datetime, end_date: datetime
    ):
        name, self.agg = layer.split("__")
        self.confirmed = "confirmed" in name

        self.start_date = start_date
        self.end_date = end_date

        super().__init__(layer, tile)

        # if only confirmed, remove any value beneath 30000 (which is unconfirmed)
        if self.confirmed:
            self.data[self.data < 30000] = 0

        # remove conf and set to ordinal date since 2015
        self.data %= 10000

        if self.start_date:
            start_ordinal = self.start_date.toordinal() - date(2014, 12, 31).toordinal()
            self.data[self.data < start_ordinal] = self.no_data_value

        if self.end_date:
            end_ordinal = self.end_date.toordinal() - date(2014, 12, 31).toordinal()
            self.data[self.data > end_ordinal] = self.no_data_value

    def get_raster_uri(self):
        # return hardcoded URL
        tile_id = self.get_tile_id()
        print(f"TILE ID: {tile_id}")
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
        return super().result

    @result.setter
    def result(self, value):
        # value is already ordinal date sine 2015, so just need to add 2015 ordinal date to get iso date
        value += date(2014, 12, 31).toordinal()

        if self.agg == "isoweek":
            # change to ordinal date of beginning of iso week
            value = [date.fromordinal(ordinal) for ordinal in value]
            value = [
                (d - timedelta(days=d.isoweekday() - 1)).toordinal() for d in value
            ]

        self._result = np.array(value)

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
