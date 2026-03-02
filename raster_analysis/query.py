# flake8: noqa
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mo_parsing.results import ParseResults
from mo_sql_parsing import parse
from numpy import ndarray
from pydantic import BaseModel

from raster_analysis.data_environment import (
    DataEnvironment,
    DerivedLayer,
    Layer,
    SourceLayer,
)
from raster_analysis.exceptions import QueryParseException
from raster_analysis.grid import Grid, GridName
from raster_analysis.window import SourceWindow

# After upgrading to Python 3.11, this can become just
# from enum import StrEnum
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass


class SpecialSelectors(StrEnum):
    latitude = "latitude"
    longitude = "longitude"
    area__ha = "area__ha"


class ComparisonOperator(StrEnum):
    gt = ">"
    lt = "<"
    gte = ">="
    lte = "<="
    eq = "=="
    neq = "!="


class Sort(StrEnum):
    asc = "asc"
    desc = "desc"


class SetOperator(StrEnum):
    intersect = "intersect"
    union = "union"


class Filter:
    """Base class representing an empty filter."""

    def apply(self, tile_width: int, windows: Dict[str, ndarray]):
        return np.ones((tile_width, tile_width))

    def get_layers(self):
        return []


class FilterLeaf(Filter):
    def __init__(self, layer, operator: ComparisonOperator, value):
        self.layer = layer
        self.operator = operator
        self.value = value

    def apply(self, tile_width: int, windows: Dict[str, SourceWindow]) -> ndarray:
        window = windows[self.layer].data
        return eval(f"window {self.operator.value} self.value")

    def get_layers(self):
        return [self.layer]


class FilterNode(Filter):
    def __init__(self, filters: Sequence[Filter], operator: SetOperator):
        self.filters = filters
        self.operator = operator

    def apply(self, tile_width: int, windows: Dict[str, ndarray]) -> ndarray:
        if self.operator == SetOperator.intersect:
            node_filter = np.ones((tile_width, tile_width))
            for f in self.filters:
                node_filter *= f.apply(tile_width, windows)
        elif self.operator == SetOperator.union:
            node_filter = np.zeros((tile_width, tile_width)).astype(dtype=bool)
            for f in self.filters:
                node_filter |= f.apply(tile_width, windows)
        else:
            raise ValueError(f"Set operator {self.operator} not implemented.")

        return node_filter

    def get_layers(self) -> List[Layer]:
        layers: List[Layer] = []
        for f in self.filters:
            layers += f.get_layers()

        return layers


class SupportedAggregates(StrEnum):
    sum = "sum"
    avg = "avg"
    count_ = "count"


class Aggregate(BaseModel):
    name: SupportedAggregates
    layer: str
    alias: Optional[str] = None


class Function(StrEnum):
    isoweek = "isoweek"


class Selector(BaseModel):
    layer: str
    function: Optional[Function] = None
    alias: Optional[str] = None

    def __hash__(self):
        return hash(self.layer)


class Query:
    def __init__(self, query: str, data_environment: DataEnvironment):
        self.data_environment = data_environment
        (
            base,
            selectors,
            filter,
            groups,
            aggregates,
            order_by,
            sort,
            limit,
        ) = self.parse_query(query)

        self.raw_query = query
        self.base = base
        self.selectors = selectors
        self.filter = filter
        self.groups = groups
        self.aggregates = aggregates
        self.order_by = order_by
        self.sort = sort
        self.limit = limit

        self.validate_query()

    def validate_query(self):
        layer_names = self.get_layer_names()
        self.data_environment.get_layers(layer_names)

    def get_source_layers(self) -> List[SourceLayer]:
        layer_names = self.get_layer_names()
        return self.data_environment.get_source_layers(layer_names)

    def get_derived_layers(self) -> List[DerivedLayer]:
        layer_names: List[str] = self.get_layer_names()
        return self.data_environment.get_derived_layers(layer_names)

    def get_layer_names(self) -> List[str]:
        layers = [selector.layer for selector in self.selectors]
        layers += self.filter.get_layers()
        layers += [
            agg.layer
            for agg in self.aggregates
            if agg.name != SupportedAggregates.count_
        ]
        layers += [group.layer for group in self.groups]

        if self.data_environment.has_layer(self.base.layer):
            layers.append(self.base.layer)

        return list(dict.fromkeys(layers))

    def get_result_selectors(self) -> List[Selector]:
        layers = [selector for selector in self.selectors]
        layers += [group for group in self.groups]

        return list(dict.fromkeys(layers))

    def get_group_columns(self) -> List[str]:
        return [group.layer for group in self.groups]

    def get_order_by_columns(self) -> List[str]:
        return [order_by.layer for order_by in self.order_by]

    def get_minimum_grid(self) -> Grid:
        layers: List[SourceLayer] = self.get_source_layers()
        grids: List[Grid] = [
            self.data_environment.get_layer_grid(layer.name) for layer in layers
        ]

        if grids:
            minimum_grid = grids[0]
            for grid in grids:
                if grid.get_pixel_width() < minimum_grid.get_pixel_width():
                    minimum_grid = grid
        else:
            minimum_grid = Grid.get_grid(GridName.ten_by_forty_thousand)

        return minimum_grid

    def parse_query(self, raw_query: str):
        parsed: ParseResults = parse(raw_query)

        if "select" not in parsed or "from" not in parsed:
            raise QueryParseException(
                "Invalid query, must include SELECT and FROM components"
            )

        base = Selector(layer=parsed["from"])
        selectors, aggregates = self._parse_select(parsed)
        where = self._parse_where(parsed)
        groups = self._parse_group_by(parsed)
        order_by, sort = self._parse_order_by(parsed)
        limit = parsed.get("limit", None)

        return base, selectors, where, groups, aggregates, order_by, sort, limit

    def _parse_select(
        self, query: ParseResults
    ) -> Tuple[List[Selector], List[Aggregate]]:
        selectors: List[Selector] = []
        aggregates: List[Aggregate] = []
        for selector in Query._ensure_list(query["select"]):
            alias = selector.get("name", None)
            if isinstance(selector["value"], dict):
                func_name, layer_name = Query._get_first_key_value(selector["value"])
                if func_name in SupportedAggregates.__members__.values():
                    aggregate = Aggregate(
                        name=SupportedAggregates(func_name),
                        layer=layer_name,
                        alias=alias,
                    )
                    aggregates.append(aggregate)
                elif func_name in Function.__members__.values():
                    selector = Selector(
                        layer=layer_name, function=Function(func_name), alias=alias
                    )
                    selectors.append(selector)
            elif isinstance(selector["value"], str):
                selector = Selector(layer=selector["value"], alias=alias)
                selectors.append(selector)

        return selectors, aggregates

    def _parse_where(self, query: ParseResults) -> Filter:
        if "where" in query:
            return self._parse_filter(query["where"])

        return Filter()

    def _parse_filter(self, filter) -> FilterNode:
        op, values = self._get_first_key_value(filter)
        if op in ["and", "or"]:
            filters = [self._parse_filter(value) for value in values]
            return FilterNode(filters, self.get_set_operator(op))
        elif op in ComparisonOperator.__members__:
            layer, value = values
            if isinstance(value, dict):
                value = value["literal"]

            encoded_values = self.data_environment.encode_layer(layer, value)
            return FilterNode(
                [
                    FilterLeaf(layer, ComparisonOperator[op], enc_val)
                    for enc_val in encoded_values
                ],
                SetOperator.union,
            )
        raise QueryParseException(f"Unsupported filter operator: {op}")

    def _parse_group_by(self, query: ParseResults) -> List[Selector]:
        groups: List[Selector] = []
        if "groupby" in query:
            for group in Query._ensure_list(query["groupby"]):
                if isinstance(group["value"], dict):
                    func_name, layer_name = Query._get_first_key_value(group["value"])
                    if func_name in Function.__members__.values():
                        group = Selector(layer=layer_name, function=Function(func_name))
                        groups.append(group)
                    else:
                        raise QueryParseException(
                            f"Unsupported function {func_name} for selector {layer_name} in GROUP BY"
                        )
                elif isinstance(group["value"], str):
                    group = Selector(layer=group["value"])
                    groups.append(group)

        return groups

    def _parse_order_by(self, query: ParseResults) -> Tuple[List[Selector], Any]:
        order_bys: List[Selector] = []
        sort = Sort.asc

        if "orderby" in query:
            for order_by in Query._ensure_list(query["orderby"]):
                order_bys.append(Selector(layer=order_by["value"]))

                if "sort" in order_by:
                    sort = order_by["sort"].lower()

        return order_bys, sort

    # the SQL parser sometimes outputs a list or single value depending on input,
    # Just a helper to make it consistent
    @staticmethod
    def _ensure_list(a: Union[List, Any]) -> List:
        return a if isinstance(a, List) else [a]

    # certain things are parsed single key value pairs, so just get the first key value pair
    @staticmethod
    def _get_first_key_value(d: Dict[str, Any]) -> Tuple[str, Any]:
        for key, val in d.items():
            return (key, val)
        raise Exception("Should we be here?")

    @staticmethod
    def get_set_operator(sql_op: str) -> SetOperator:
        return {
            "and": SetOperator.intersect,
            "or": SetOperator.union,
        }[sql_op]
