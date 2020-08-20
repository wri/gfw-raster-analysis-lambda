import numpy as np
from rasterio import Affine
from shapely.geometry import Polygon
from numpy import ndarray
from math import floor

from raster_analysis.grid import get_raster_uri
from raster_analysis.io import read_window

from datetime import date, timedelta, datetime
from typing import Tuple, Optional

from raster_analysis.globals import Numeric, WINDOW_SIZE, DATA_LAKE_LAYER_MANAGER
from .layer import LayerInfo


class Window:
    def __init__(self, layer: str, tile: Polygon):
        self.layer: LayerInfo = LayerInfo(layer)
        self.tile: Polygon = tile

        data, shifted_affine, no_data_value = self.read(tile)
        self.data: ndarray = data
        self.shifted_affine: Affine = shifted_affine
        self.no_data_value: Numeric = no_data_value

        self._result: Optional[ndarray] = None

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
        return self._result.tolist()

    @result.setter
    def result(self, value):
        self._result = value

    @property
    def result_col_name(self):
        return self.layer.name_type

    def get_raster_uri(self):
        return get_raster_uri(self.layer.name, self.layer.data_type, self.tile)

    def has_default_value(self):
        return False


def get_window(
    layer: str, tile: Polygon, start_date: datetime, end_date: datetime
) -> Window:
    if layer == "area__ha":
        return AreaWindow(layer, tile)
    elif layer == "alert__count":
        return CountWindow(layer, tile)
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
        return DataLakeWindow(layer, tile)


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


class DataLakeWindow(Window):
    @property
    def result(self):
        return super().result

    @result.setter
    def result(self, value):
        self._result = np.array(
            [
                DATA_LAKE_LAYER_MANAGER.get_layer_value(self.layer.name, val)
                for val in value
            ]
        )

    def has_default_value(self):
        return DATA_LAKE_LAYER_MANAGER.has_default_value(
            self.layer.name, self.no_data_value
        )


class TcdWindow(DataLakeWindow):
    def __init__(self, layer: str, tile: Polygon):
        name, threshold = layer.split("__")
        self.threshold = int(threshold)

        super().__init__(f"{name}__threshold", tile)
        threshold_pixel_value = DATA_LAKE_LAYER_MANAGER.get_pixel_value(
            name, self.threshold
        )

        self.data = self.data >= threshold_pixel_value


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
