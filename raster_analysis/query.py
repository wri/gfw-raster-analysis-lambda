from enum import Enum
from typing import Any, Dict, List, Optional

from moz_sql_parser import parse
from numpy import ndarray
from pydantic import BaseModel

from raster_analysis.data_environment import DataEnvironment, Layer
from raster_analysis.exceptions import QueryParseException
from raster_analysis.grid import Grid, GridName


class SpecialSelectors(str, Enum):
    latitude = "latitude"
    longitude = "longitude"
    area__ha = "area__ha"


class Operator(str, Enum):
    gt = ">"
    lt = "<"
    gte = ">="
    lte = "<="
    eq = "=="
    neq = "!="


class Filter(BaseModel):
    operator: Operator
    layer: str
    value: Any

    def apply_filter(self, window: ndarray) -> ndarray:
        return eval(f"window {self.operator.value} self.value")


class SupportedAggregates(str, Enum):
    sum = "sum"
    avg = "avg"
    count_ = "count"


class Aggregate(BaseModel):
    name: SupportedAggregates
    layer: str


class Function(str, Enum):
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
        base, selectors, filters, groups, aggregates = self.parse_query(query)

        self.base = base
        self.selectors = selectors
        self.filters = filters
        self.groups = groups
        self.aggregates = aggregates

        self.validate_query()

    def validate_query(self):
        layer_names = self.get_layer_names()
        self.data_environment.get_layers(layer_names)

    def get_source_layers(self) -> List[Layer]:
        layer_names = self.get_layer_names()
        return self.data_environment.get_source_layers(layer_names)

    def get_derived_layers(self) -> List[Layer]:
        layer_names = self.get_layer_names()
        return self.data_environment.get_derived_layers(layer_names)

    def get_layer_names(self) -> List[str]:
        layers = [selector.layer for selector in self.selectors]
        layers += [filter.layer for filter in self.filters]
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

    def get_minimum_grid(self) -> Grid:
        layers = self.get_source_layers()
        grids = [self.data_environment.get_layer_grid(layer.name) for layer in layers]

        if grids:
            minimum_grid = grids[0]
            for grid in grids:
                if grid.get_pixel_width() < minimum_grid.get_pixel_width():
                    minimum_grid = grid
        else:
            minimum_grid = Grid.get_grid(GridName.ten_by_forty_thousand)

        return minimum_grid

    def parse_query(self, raw_query: str):
        parsed = parse(raw_query)

        if "select" not in parsed or "from" not in parsed:
            raise QueryParseException(
                "Invalid query, must include SELECT and FROM components"
            )

        base = Selector(layer=parsed["from"])
        selectors, aggregates = self._parse_select(parsed)
        where = self._parse_where(parsed)
        groups = self._parse_group_by(parsed)

        return base, selectors, where, groups, aggregates

    def _parse_select(self, query: Dict[str, Any]):
        selectors = []
        aggregates = []
        for selector in Query._ensure_list(query["select"]):
            if isinstance(selector["value"], dict):
                func_name, layer_name = Query._get_first_key_value(selector["value"])
                if func_name in SupportedAggregates.__members__.values():
                    aggregate = Aggregate(name=func_name, layer=layer_name)
                    aggregates.append(aggregate)
                elif func_name in Function.__members__.values():
                    selector = Selector(layer=layer_name, function=func_name)
                    selectors.append(selector)
            elif isinstance(selector["value"], str):
                selector = Selector(layer=selector["value"])
                selectors.append(selector)

        return selectors, aggregates

    def _parse_where(self, query: Dict[str, Any]):
        where = []
        if "where" in query:
            if "or" in query["where"]:
                raise QueryParseException("OR statement is not supported.")
            elif "and" in query["where"]:
                if isinstance(query["where"]["and"], dict):
                    raise QueryParseException(
                        "Only one level is supported in AND statement."
                    )
                filters = query["where"]["and"]
            else:
                filters = query["where"]

            for filter in Query._ensure_list(filters):
                op, (layer, value) = Query._get_first_key_value(filter)
                if isinstance(value, dict):
                    value = value["literal"]

                encoded_values = self.data_environment.encode_layer(layer, value)
                where += [
                    Filter(operator=Operator[op], layer=layer, value=enc_val)
                    for enc_val in encoded_values
                ]

        return where

    def _parse_group_by(self, query: Dict[str, Any]):
        groups = []
        if "groupby" in query:
            for group in Query._ensure_list(query["groupby"]):
                if isinstance(group["value"], dict):
                    func_name, layer_name = Query._get_first_key_value(group["value"])
                    if func_name in Function.__members__.values():
                        group = Selector(layer=layer_name, function=func_name)
                        groups.append(group)
                    else:
                        raise QueryParseException(
                            f"Unsupported function {func_name} for selector {layer_name} in GROUP BY"
                        )
                elif isinstance(group["value"], str):
                    group = Selector(layer=group["value"])
                    groups.append(group)

        return groups

    # the SQL parser sometimes outputs a list or single value depending on input,
    # Just a helper to make it consistent
    @staticmethod
    def _ensure_list(a):
        return a if type(a) is list else [a]

    # certain things are parsed single key value pairs, so just get the first key value pair
    @staticmethod
    def _get_first_key_value(d: Dict[str, str]):
        for key, val in d.items():
            return (key, val)
