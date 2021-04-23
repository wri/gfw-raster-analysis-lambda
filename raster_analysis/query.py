from typing import List, Any, Dict
from enum import Enum

from numpy import ndarray
from pydantic import BaseModel
from moz_sql_parser import parse

from raster_analysis.data_environment import DataEnvironment, Layer
from raster_analysis.grid import Grid
from raster_analysis.exceptions import QueryParseException


class SpecialSelectors(str, Enum):
    latitude = "latitude"
    longitude = "longitude"
    area__ha = "area__ha"
    alert__count = "alert__count"  # deprecated


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
    function: Function = None
    alias: str = None


class Query(BaseModel):
    base: Selector
    selectors: List[Selector]
    filters: List[Filter] = []
    groups: List[Selector] = []
    aggregates: List[Aggregate] = []
    data_environment: DataEnvironment

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
        layers += [group for group in self.groups]

        if self.has_layer(self.base.layer):
            layers.append(self.base.layer)

        return list(dict.fromkeys(layers))

    def get_result_selectors(self) -> List[Selector]:
        layers = [selector for selector in self.selectors]
        layers += [aggregate.layer for aggregate in self.aggregates]
        layers += [group for group in self.groups]

        return list(dict.fromkeys(layers))

    def get_group_columns(self) -> List[str]:
        return [group.layer for group in self.groups]

    def get_minimum_grid(self) -> Grid:
        layers = self.get_layer_names()
        grids = [self.data_environment.get_layer_grid(layer.name) for layer in layers]

        minimum_grid = grids[0]
        for grid in grids:
            if grid.get_pixel_width() < minimum_grid.get_pixel_width():
                minimum_grid = grid

        return minimum_grid

    @staticmethod
    def parse_query(raw_query: str, data_environment: DataEnvironment):
        parsed = parse(raw_query)
        selectors = []
        where = []
        groups = []
        aggregates = []

        if "select" not in parsed or "from" not in parsed:
            raise QueryParseException(
                "Invalid query, must include SELECT and FROM components"
            )

        base = data_environment.get_layer(parsed["from"])
        for selector in Query._ensure_list(parsed["select"]):
            if isinstance(selector["value"], dict):
                func_name, layer_name = Query._get_first_key_value(selector["value"])
                if func_name in SupportedAggregates.__members__:
                    aggregate = Aggregate(name=func_name, layer=layer_name)
                    aggregates.append(aggregate)
                elif func_name in Function.__members__:
                    selector = Selector(layer=layer_name, function=func_name)
                    selectors.append(selector)
            elif isinstance(selector["value"], str):
                selector = Selector(layer=selector["value"], function=func_name)
                selectors.append(selector)

        if "where" in parsed:
            if "and" in parsed["where"]:
                if isinstance(parsed["where"]["and"], dict):
                    raise QueryParseException(
                        "Only one level is supported in AND statement."
                    )
                filters = parsed["where"]["and"]
            elif "or" in parsed["where"]:
                raise QueryParseException("OR statement is not supported.")
            else:
                filters = parsed["where"]

            for filter in Query._ensure_list(filters):
                op, (layer, value) = Query._get_first_key_value(filter)
                if isinstance(value, dict):
                    value = value["literal"]

                layer = data_environment.get_layer(layer)
                if layer.pixel_encoding:
                    # TODO reverse pixel encoding?
                    for enc_val in layer.encoder(value):
                        where.append(
                            Filter(operator=Operator[op], layer=layer, value=enc_val)
                        )
                else:
                    where.append(
                        Filter(operator=Operator[op], layer=layer, value=value)
                    )

        if "groupby" in parsed:
            for group in Query._ensure_list(parsed["groupby"]):
                if isinstance(group["value"], dict):
                    func_name, layer_name = Query._get_first_key_value(
                        selector["value"]
                    )
                    if func_name in Function.__members__:
                        group = Selector(layer=layer_name, function=func_name)
                        groups.append(group)
                    else:
                        raise ValueError(
                            f"Unsupported function `func_name` for selector `layer_name` in GROUP BY"
                        )
                elif isinstance(group["value"], str):
                    group = Selector(layer=group["value"], function=func_name)
                    groups.append(group)

                groups.append(data_environment.get_layer(group["value"]))

        return Query(
            base=base,
            selectors=selectors,
            filters=where,
            groups=groups,
            aggregates=aggregates,
            data_environment=data_environment,
        )

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
