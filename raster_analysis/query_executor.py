from typing import List, Any, Union, Tuple
from io import StringIO

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from rasterio.transform import xy

from raster_analysis.data_cube import DataCube
from raster_analysis.query import Query, AggregateFunction, SpecialSelectors, Aggregate


class QueryExecutor:
    def __init__(
        self,
        query: Query,
        data_cube: DataCube
    ):
        self.query = query
        self.data_cube = data_cube

    def execute(self):
        mask = self.data_cube.mask

        for filter in self.query.filters:
            window = self.data_cube.windows[filter.layer]
            mask *= filter.apply_filter(window.data)

        if self.query.aggregates:
            self.result = self._aggregate(mask)
        elif self.query.selectors:
            self.result = self._select(mask)

    def result_as_csv(self) -> StringIO:
        buffer = StringIO()
        self.result.to_csv(buffer, index=False)
        return buffer

    def _aggregate(self, mask: ndarray) -> DataFrame:
        if self.query.groups:
            return self._aggregate_by_group(mask)
        else:
            return self._aggregate_all(mask)

    def _aggregate_by_group(self, mask: ndarray) -> DataFrame:
        group_windows = [self.data_cube.windows[layer] for layer in self.query.groups]
        for window in group_windows:
            mask *= window.data.astype(dtype=np.bool)

        group_columns = [np.ravel(window.data) for window in group_windows]
        group_dimensions = [col.max() + 1 for col in group_columns]

        # numpy trick to consolidate unique combinations of group values into a single number
        # running np.unique is way faster on single numbers than arrays
        group_multi_index = np.ravel_multi_index(group_columns, group_dimensions).astype(np.uint32)
        if mask is not None:
            group_multi_index = np.compress(np.ravel(mask), group_multi_index)

        # the inverse index contains the indices of each unique value
        # this can be used later to calculate sums of each group
        group_indices, inverse_index, group_counts = np.unique(
            group_multi_index, return_counts=True, return_inverse=True
        )

        group_column_names = [group.layer for group in self.query.groups]
        agg_column_names = [agg.layer.layer for agg in self.query.aggregates]

        results = dict(zip(group_column_names, np.unravel_index(group_indices, group_dimensions)))

        agg_columns = [
            self._aggregate_window_by_group(agg, mask, group_counts, inverse_index)
            for agg in self.query.aggregates
        ]
        results.update(dict(zip(agg_column_names, agg_columns)))

        return pd.DataFrame(results)

    def _aggregate_window_by_group(
            self, aggregate: Aggregate, mask: ndarray, group_counts: List[int], inverse_index: List[int]
    ) -> ndarray:
        if aggregate.layer.layer == SpecialSelectors.count:
            return group_counts
        elif aggregate.layer.layer == SpecialSelectors.area:
            return group_counts * self.data_cube.mean_area
        else:
            window = self.data_cube.windows[aggregate.layer]
            masked_data = np.extract(mask, window.data)

            # this will sum the values of aggregate data into different bins, where each bin
            # is the corresponding group at that pixel
            sums = np.bincount(inverse_index, weights=masked_data, minlength=group_counts.size)
            if aggregate.function == AggregateFunction.sum:
                if aggregate.layer.is_area_density:
                    # layer value representing area density need to be multiplied by area to get gross value
                    return sums * self.data_cube.mean_area
                return sums
            elif aggregate.function == AggregateFunction.avg:
                return sums / masked_data.size

    def _aggregate_all(self, mask: ndarray) -> DataFrame:
        results = {}

        for agg in self.query.aggregates:
            results[agg.layer.layer] = [self._aggregate_window(agg, mask)]

        return pd.DataFrame(results)

    def _aggregate_window(self, aggregate: Aggregate, mask: ndarray) -> Union[int, float]:
        if aggregate.layer.layer == SpecialSelectors.count:
            return mask.sum()
        elif aggregate.layer.layer == SpecialSelectors.area:
            return mask.sum() * self.data_cube.mean_area
        else:
            window = self.data_cube.windows[aggregate.layer]
            masked_data = np.extract(mask, window.data)
            sum = masked_data.sum()

            if aggregate.function == AggregateFunction.sum:
                if aggregate.layer.is_area_density:
                    return sum * self.data_cube.mean_area
                return sum
            elif aggregate.function == AggregateFunction.avg:
                return sum / masked_data.size

    def _select(self, mask: ndarray) -> DataFrame:
        results = {}

        if SpecialSelectors.latitude in self.query.selectors or SpecialSelectors.longitude in self.query.selectors:
            latitudes, longitudes = self._extract_coordinates(mask)
            results[SpecialSelectors.latitude.value] = latitudes
            results[SpecialSelectors.longitude.value] = longitudes

        for selector in self.query.selectors:
            window = self.data_cube.windows[selector.layer].window.data
            window *= mask
            values = np.extract(window != 0, window)
            results[selector.layer] = values

        return pd.DataFrame(results)

    def _extract_coordinates(self, mask: ndarray) -> List[Tuple[float, float]]:
        rows, cols = np.nonzero(mask)
        return xy(self.data_cube.shifted_affine, rows, cols)
