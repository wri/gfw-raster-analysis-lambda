from typing import List, Any, Union, Tuple
from io import StringIO

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from rasterio.transform import xy

from raster_analysis.data_cube import DataCube
from raster_analysis.data_lake import LAYERS
from raster_analysis.query import Query, AggregateFunction, SpecialSelectors, Aggregate


class QueryExecutor:
    def __init__(self, query: Query, data_cube: DataCube):
        self.query = query
        self.data_cube = data_cube

    def execute(self) -> DataFrame:
        mask = self.data_cube.mask

        if self.query.base in self.data_cube.windows:
            mask *= self.data_cube.windows[self.query.base].data.astype(dtype=np.bool)

        for filter in self.query.filters:
            window = self.data_cube.windows[filter.layer]
            mask *= filter.apply_filter(window.data)

        if self.query.aggregates:
            return self._aggregate(mask)
        elif self.query.selectors:
            return self._select(mask)

    def _aggregate(self, mask: ndarray) -> DataFrame:
        if self.query.groups:
            return self._aggregate_by_group(mask)
        else:
            return self._aggregate_all(mask)

    def _aggregate_by_group(self, mask: ndarray) -> DataFrame:
        group_windows = []
        for layer in self.query.groups:
            window = self.data_cube.windows[layer]
            group_windows.append(self.data_cube.windows[layer])

            if not layer.has_default_value:
                mask *= window.data.astype(dtype=np.bool)

        group_columns = [np.ravel(window.data) for window in group_windows]
        group_dimensions = [col.max() + 1 for col in group_columns]

        # numpy trick to consolidate unique combinations of group values into a single number
        # running np.unique is way faster on single numbers than arrays
        group_multi_index = np.ravel_multi_index(
            group_columns, group_dimensions
        ).astype(np.uint32)
        if mask is not None:
            group_multi_index = np.compress(np.ravel(mask), group_multi_index)

        # the inverse index contains the indices of each unique value
        # this can be used later to calculate sums of each group
        group_indices, inverse_index, group_counts = np.unique(
            group_multi_index, return_counts=True, return_inverse=True
        )

        group_column_names = [group.layer for group in self.query.groups]
        results = dict(
            zip(group_column_names, np.unravel_index(group_indices, group_dimensions))
        )

        for agg in self.query.aggregates:
            column_name, column_val = self._aggregate_window_by_group(
                agg, mask, group_counts, inverse_index
            )
            results[column_name] = column_val

        return pd.DataFrame(results)

    def _aggregate_window_by_group(
        self,
        aggregate: Aggregate,
        mask: ndarray,
        group_counts: ndarray,
        inverse_index: ndarray,
    ) -> Tuple[str, ndarray]:
        if aggregate.layer.layer == SpecialSelectors.alert__count:
            return SpecialSelectors.alert__count, group_counts
        elif aggregate.function == AggregateFunction.count_:
            return AggregateFunction.count_, group_counts
        elif aggregate.layer.layer == SpecialSelectors.area__ha:
            return SpecialSelectors.area__ha, group_counts * self.data_cube.mean_area
        else:
            window = self.data_cube.windows[aggregate.layer]
            masked_data = np.extract(mask, window.data)
            # column_name = f"sum({aggregate.layer.layer})"

            # this will sum the values of aggregate data into different bins, where each bin
            # is the corresponding group at that pixel
            sums = np.bincount(
                inverse_index, weights=masked_data, minlength=group_counts.size
            )
            if aggregate.function == AggregateFunction.sum:
                if aggregate.layer.is_area_density:
                    # layer value representing area density need to be multiplied by area to get gross value
                    return aggregate.layer.layer, sums * self.data_cube.mean_area
                return aggregate.layer.layer, sums
            elif aggregate.function == AggregateFunction.avg:
                return aggregate.layer.layer, sums / masked_data.size
            else:
                raise NotImplementedError("Undefined aggregate function")

    def _aggregate_all(self, mask: ndarray) -> DataFrame:
        results = {}

        for agg in self.query.aggregates:
            column_name, column_val = self._aggregate_window(agg, mask)
            results[column_name] = [column_val]

        return pd.DataFrame(results)

    def _aggregate_window(
        self, aggregate: Aggregate, mask: ndarray
    ) -> Tuple[str, Union[int, float]]:
        if aggregate.layer.layer == SpecialSelectors.alert__count:
            return SpecialSelectors.alert__count, mask.sum()
        elif aggregate.function == AggregateFunction.count_:
            return AggregateFunction.count_, mask.sum()
        elif aggregate.layer.layer == SpecialSelectors.area__ha:
            return aggregate.layer.layer, mask.sum() * self.data_cube.mean_area
        else:
            window = self.data_cube.windows[aggregate.layer]
            masked_data = np.extract(mask, window.data)
            sum = masked_data.sum()

            if aggregate.function == AggregateFunction.sum:
                if aggregate.layer.is_area_density:
                    return aggregate.layer.layer, sum * self.data_cube.mean_area
                return aggregate.layer.layer, sum
            elif aggregate.function == AggregateFunction.avg:
                return aggregate.layer.layer, sum / masked_data.size
            else:
                raise NotImplementedError("Undefined aggregate function")

    def _select(self, mask: ndarray) -> DataFrame:
        results = {}

        selector_names = [
            selector.alias if selector.alias else selector.layer
            for selector in self.query.selectors
        ]

        if (
            SpecialSelectors.latitude in selector_names
            or SpecialSelectors.longitude in selector_names
        ):
            latitudes, longitudes = self._extract_coordinates(mask)
            results[SpecialSelectors.latitude.value] = np.array(latitudes).astype(
                np.double
            )
            results[SpecialSelectors.longitude.value] = np.array(longitudes).astype(
                np.double
            )
            selector_names.remove(SpecialSelectors.latitude.value)
            selector_names.remove(SpecialSelectors.longitude.value)

        for selector in selector_names:
            window = self.data_cube.windows[LAYERS[selector]].data
            values = np.extract(mask, window)
            results[selector] = values

        return pd.DataFrame(results)

    def _extract_coordinates(self, mask: ndarray) -> List[Tuple[float, float]]:
        rows, cols = np.nonzero(mask)
        return xy(self.data_cube.shifted_affine, rows, cols)
