from datetime import date, timedelta, datetime
from typing import Tuple, Optional, List, Any, Union, cast
from math import floor

import numpy as np
from rasterio import Affine
from shapely.geometry import Polygon
from numpy import ndarray
from aws_xray_sdk.core import xray_recorder

from raster_analysis.grid import get_raster_uri
from raster_analysis.io import read_window_ignore_missing
from raster_analysis.globals import (
    Numeric,
    ResultValue,
    WINDOW_SIZE,
    DATA_LAKE_LAYER_MANAGER,
)
from .layer import LayerInfo

ResultValues = Union[ndarray, ResultValue]


class Window:
    def __init__(self, layer: str, tile: Polygon):
        self.layer: LayerInfo = LayerInfo(layer)
        self.tile: Polygon = tile

        data, shifted_affine, no_data_value = self.read(tile)

        if data.size == 0:
            self.empty = True
            data = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.uint8)
        else:
            self.empty = False

        self.data: ndarray = data
        self.shifted_affine: Affine = shifted_affine
        self.no_data_value: Numeric = no_data_value

        self._result: ResultValues = np.array([])

    def read(self, tile: Polygon) -> Tuple[np.ndarray, Affine, Numeric]:
        data, shifted_affine, no_data_value = read_window_ignore_missing(
            self.get_raster_uri(), tile
        )

        return data, shifted_affine, no_data_value

    @xray_recorder.capture("Window Sum")
    def sum(
        self,
        mean_area: int,
        mask: ndarray,
        linear_index: ndarray = None,
        index_counts: ndarray = None,
    ) -> Union[Any, ndarray]:
        """
        Generic sum operation for windows based on groupings from linear index.
        :param mean_area: mean area for tile. Used if area is necessary for sum calculation.
        :param mask: A mask over the window to determine which pixels to sum.
        :param linear_index: Linear index calculated for groupings, if they exist. Otherwise whole window will be summed
            to a single number.
        :param index_counts: Counts for each value in the linear index to determine number of bins for output.
        :return: Either single sum number of 1d array of sum values per grouping in ascending order of linear index values.
        """
        data = np.extract(mask, self.data)

        if linear_index is None or index_counts is None:
            return data.sum()

        return np.bincount(linear_index, weights=data, minlength=index_counts.size)

    def clear(self) -> None:
        """
        Clear internal data array to save memory.
        """
        self.data = []

    @property
    def result(self) -> Union[ResultValue, List[ResultValue]]:
        if isinstance(self._result, ndarray):
            return self._result.tolist()
        else:
            return self._result

    @result.setter
    def result(self, value: ResultValues) -> None:
        self._result = value

    @property
    def result_col_name(self) -> str:
        return self.layer.name_type

    def get_raster_uri(self) -> str:
        return get_raster_uri(self.layer.name, self.layer.data_type, self.tile)

    def has_default_value(self) -> bool:
        """Check if NoData values should be interpreted as default value. False by default for Windows."""
        return False


def get_window(
    layer: str,
    tile: Polygon,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
) -> Window:
    layer_info = LayerInfo(layer)

    if layer_info.name_type == "area__ha":
        return AreaWindow(layer, tile)
    elif layer_info.name_type == "alert__count":
        return CountWindow(layer, tile)
    elif layer_info.name == "umd_glad_alerts":
        return GladAlertsWindow(layer, tile, start_date, end_date)
    elif layer_info.data_type == "year":
        return YearWindow(layer, tile, start_date, end_date)
    elif "umd_tree_cover_density" in layer_info.name:
        return TcdWindow(layer, tile)
    elif layer_info.data_type == "ha-1":
        return AreaDensityWindow(layer, tile)
    elif layer_info.name == "whrc_aboveground_co2_emissions":
        return CarbonEmissionsWindow(layer, tile)
    else:
        return DataLakeWindow(layer, tile)


class YearWindow(Window):
    """
    Class representing year layers, which is always encoded as (year - 2000)
    """

    @xray_recorder.capture("Initialize Year Window")
    def __init__(
        self, layer: str, tile: Polygon, start_date: datetime, end_date: datetime
    ):
        self.start_date: datetime = start_date
        self.end_date: datetime = end_date

        super().__init__(layer, tile)

        if not self.empty:
            if self.start_date:
                start_year = self.start_date.year - 2000
                self.data[self.data < start_year] = self.no_data_value

            if self.end_date:
                end_year = self.end_date.year - 2000
                self.data[self.data > end_year] = self.no_data_value

    @property
    def result(self) -> Union[ResultValue, List[ResultValue]]:
        return super().result

    @result.setter
    def result(self, value: ResultValues) -> None:
        value = cast(ndarray, value)
        self._result = value + 2000


class DataLakeWindow(Window):
    @property
    def result(self) -> Union[ResultValue, List[ResultValue]]:
        return super().result

    @result.setter
    def result(self, value: ResultValues):
        if isinstance(value, ndarray):
            value = cast(ndarray, value)
            self._result = np.array(
                [
                    DATA_LAKE_LAYER_MANAGER.get_layer_value(self.layer.name, str(val))
                    for val in value
                ]
            )
        else:
            DATA_LAKE_LAYER_MANAGER.get_layer_value(self.layer.name, str(value))

    def has_default_value(self) -> bool:
        """
        For data lake layers, keep NoData values if they have a default mapping.
        :return: True if it has a default mapping, False if it doesn't or has no mappings
        """
        return DATA_LAKE_LAYER_MANAGER.has_default_value(
            self.layer.name, self.no_data_value
        )


class TcdWindow(DataLakeWindow):
    """
    Tree cover loss density window is interpreted in a special way, where you can put a specific
    threshold value as the data type (e.g. umd_tree_cover_density_2000__30
    """

    @xray_recorder.capture("Initialize TCD Window")
    def __init__(self, layer: str, tile: Polygon):
        name, self.threshold = layer.split("__")

        super().__init__(f"{name}__threshold", tile)

        if not self.empty:
            threshold_pixel_value = int(
                DATA_LAKE_LAYER_MANAGER.get_pixel_value(name, self.threshold)
            )

            self.data = self.data >= threshold_pixel_value


class GladAlertsWindow(Window):
    """
    Glad alert tiles are not currently in the data lake, the tiles are named differently, and the values
    are encoded in a special way with both the date and the confirmation status, so they need a lot of special
    handling. Should be simplified once we move to the data lake.
    """

    @xray_recorder.capture("Initialize Glad Alerts Window")
    def __init__(
        self,
        layer: str,
        tile: Polygon,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ):
        name, self.agg = layer.split("__")
        self.confirmed: bool = "confirmed" in name

        self.start_date: Optional[datetime] = start_date
        self.end_date: Optional[datetime] = end_date

        super().__init__(layer, tile)

        # if only confirmed, remove any value beneath 30000 (which is unconfirmed)
        if not self.empty:
            if self.confirmed:
                self.data[self.data < 30000] = 0

            # remove conf and set to ordinal date since 2015
            self.data %= 10000

            if self.start_date:
                start_ordinal = (
                    self.start_date.toordinal() - date(2014, 12, 31).toordinal()
                )
                self.data[self.data < start_ordinal] = self.no_data_value

            if self.end_date:
                end_ordinal = self.end_date.toordinal() - date(2014, 12, 31).toordinal()
                self.data[self.data > end_ordinal] = self.no_data_value

    def get_raster_uri(self) -> str:
        # return hardcoded URL
        tile_id = self.get_tile_id()
        return f"s3://gfw2-data/forest_change/umd_landsat_alerts/prod/analysis/{tile_id}.tif"

    def get_tile_id(self) -> str:
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
    def get_longitude(x: int) -> str:
        if x >= 0:
            return str(x).zfill(3) + "E"
        else:
            return str(-x).zfill(3) + "W"

    @staticmethod
    def get_latitude(y: int) -> str:
        if y >= 0:
            return str(y).zfill(2) + "N"
        else:
            return str(-y).zfill(2) + "S"

    @staticmethod
    def lower_bound(y: int) -> int:
        return int(floor(y / 10) * 10)

    @staticmethod
    def upper_bound(y: int) -> int:
        if y == GladAlertsWindow.lower_bound(y):
            return int(y)
        else:
            return int((floor(y / 10) * 10) + 10)

    @property
    def result(self) -> Union[ResultValue, List[ResultValue]]:
        return super().result

    @result.setter
    def result(self, value: ResultValues) -> None:
        # value is already ordinal date sine 2015, so just need to add 2015 ordinal date to get iso date
        value = cast(ndarray, value)
        value += date(2014, 12, 31).toordinal()

        if self.agg == "isoweek":
            # change to ordinal date of beginning of iso week
            value = [date.fromordinal(ordinal) for ordinal in value]
            value = [
                (d - timedelta(days=d.isoweekday() - 1)).toordinal() for d in value
            ]

        self._result = np.array(value)


class CountWindow(Window):
    """
    Special value `count` with no data, just overrides the sum function to return counts of each grouping or total
    """

    def __init__(self, layer: str, tile: Polygon):
        self.layer: LayerInfo = LayerInfo(layer)
        self.tile: Polygon = tile
        self.data = None

    @xray_recorder.capture("Sum Counts")
    def sum(
        self, mean_area, mask, linear_index=None, index_counts=None
    ) -> Union[int, ndarray]:
        if linear_index is None or index_counts is None:
            return mask.sum()

        return index_counts


class AreaWindow(CountWindow):
    """
    Special value `area__ha` with no data, just overrides the sum function to return the area of each grouping or total
    """

    @xray_recorder.capture("Sum Area")
    def sum(
        self, mean_area, mask, linear_index=None, index_counts=None
    ) -> Union[float, ndarray]:
        if linear_index is None or index_counts is None:
            return mask.sum() * mean_area

        return index_counts * mean_area


class AreaDensityWindow(Window):
    """
    For windows ha-1 data type, need to be multipled by area to get sum values
    """

    @xray_recorder.capture("Initialize Year Window")
    def sum(
        self, mean_area, mask, linear_index=None, index_counts=None
    ) -> Union[float, ndarray]:
        self.data = self.data.astype(np.float32) * mean_area
        return super().sum(mean_area, mask, linear_index, index_counts)


class CarbonEmissionsWindow(Window):
    """
    A virtual layer representing CO2 emissions. Based on biomass stock, multiplied by a coefficient to convert
    to CO2 quantity.
    """

    CO2_FACTOR = 0.5 * 44 / 12  # used to calculate emissions from biomass layer
    BIOMASS_LAYER = "whrc_aboveground_biomass_stock_2000"
    BIOMASS_TYPE = "Mg_ha-1"

    @xray_recorder.capture("Sum Carbon Emissions")
    def sum(self, mean_area, mask, linear_index=None, index_counts=None):
        """
        For sum, calculate gross biomass and multiply by CO2 constant.
        """
        self.data = self.data.astype(np.float32) * self.CO2_FACTOR * mean_area
        return super().sum(mean_area, mask, linear_index, index_counts)

    def get_raster_uri(self) -> str:
        return get_raster_uri(self.BIOMASS_LAYER, self.BIOMASS_TYPE, self.tile)
