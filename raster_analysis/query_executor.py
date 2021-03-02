from typing import List, Any, Union, Tuple
from io import StringIO

import numpy as np
import pandas as pd
from numpy import ndarray
from pydantic import BaseModel
from rasterio.transform import xy

from raster_analysis.data_cube import DataCube
from raster_analysis.globals import CO2_FACTOR
from raster_analysis.query import Query, AggregateFunction, SpecialSelectors, Aggregate


class QueryResult(BaseModel):
    class Column(BaseModel):
        name: str
        values: List[Any]

    columns: List[Column]


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
            window = self.windows[filter.layer]
            mask *= filter.apply_filter(window)

        if self.query.aggregates:
            self.result = self._aggregate(mask)
        elif self.query.selectors:
            self.result = self._select(mask)

    def result_as_csv(self) -> StringIO:
        buffer = StringIO()
        self.result.to_csv(buffer, index=False)
        return buffer

    def _aggregate(self, mask: ndarray) -> QueryResult:
        if self.query.groups:
            self._aggregate_by_group(mask)
        else:
            self._aggregate_all(mask)

    def _aggregate_by_group(self, mask: ndarray) -> QueryResult:
        group_windows = [self.data_cube[layer].window for layer in self.query.groups]
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

        group_column_names = [group.name for group in self.query.groups]
        agg_column_names = [agg.layer.name for agg in self.query.aggregates]

        results = dict(zip(group_column_names, np.unravel_index(group_indices, group_dimensions)))
        results.update(dict(zip(agg_column_names, np.unravel_index(group_indices, group_dimensions))))

        return pd.DataFrame(results)


    def _aggregate_window_by_group(
            self, aggregate: Aggregate, mask: ndarray, group_counts: List[int], inverse_index: List[int]
    ) -> ndarray:
        if aggregate.layer == SpecialSelectors.count:
            return group_counts
        elif aggregate.layer == SpecialSelectors.area:
            return group_counts * self.data_cube.mean_area
        else:
            window = self.data_cube[aggregate.layer]
            masked_data = np.extract(mask, window.data)

            # this will sum the values of aggregate data into different bins, where each bin
            # is the corresponding group at that pixel
            sums = np.bincount(inverse_index, weights=masked_data, minlength=group_counts.size)
            if aggregate.function == AggregateFunction.sum:
                if aggregate.layer.is_area_density:
                    # layer value representing area density need to be multiplied by area to get gross value
                    return sums * self.data_cube.mean_area
                elif aggregate.layer.is_emissions:
                    # emissions are just based on density layers multiplied by a constant
                    return sums * CO2_FACTOR * self.data_cube.mean_area
                return sums
            elif aggregate.function == AggregateFunction.avg:
                return sums / masked_data.size

    def _aggregate_all(self, mask: ndarray) -> QueryResult:
        aggregations = [
            QueryResult.Column(
                name=agg.layer.name,
                values=self._aggregate_window(agg, mask)
            )
            for agg in self.query.aggregates
        ]

        return QueryResult(aggregations)

    def _aggregate_window(self, aggregate: Aggregate, mask: ndarray) -> Union[int, float]:
        if aggregate.layer == SpecialSelectors.count:
            return mask.sum()
        elif aggregate.layer == SpecialSelectors.area:
            return mask.sum() * self.data_cube.mean_area
        else:
            window = self.data_cube[aggregate.layer]
            masked_data = np.extract(mask, window.data)
            sum = masked_data.sum()

            if aggregate.function == AggregateFunction.sum:
                if aggregate.layer.is_area_density:
                    return sum * self.data_cube.mean_area
                elif aggregate.layer.is_emissions:
                    return sum * CO2_FACTOR * self.data_cube.mean_area
                return sum
            elif aggregate.function == AggregateFunction.avg:
                return sum / masked_data.size

    def _select(self, mask: ndarray) -> QueryResult:
        columns = []

        if SpecialSelectors.latitude in self.query.selectors or SpecialSelectors.longitude in self.query.selectors:
            latitudes, longitudes = self._extract_coordinates(mask)
            columns.append(QueryResult.Column(name=SpecialSelectors.latitude.value, values=latitudes))
            columns.append(QueryResult.Column(name=SpecialSelectors.longitude.value, values=longitudes))

        for selector in self.query.selectors:
            window = self.data_cube[selector.layer].window
            window *= mask
            values = np.extract(window != 0, window)
            columns.append(QueryResult.Column(name=selector.name, values=values))

        return QueryResult(columns=columns)

    def _extract_coordinates(self, mask: ndarray) -> List[Tuple[float, float]]:
        rows, cols = np.nonzero(mask)
        return xy(self.data_cube.shifted_affine, rows, cols)
