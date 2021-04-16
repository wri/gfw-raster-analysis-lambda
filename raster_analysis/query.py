from typing import List, Any, Dict, Union
from enum import Enum

from numpy import ndarray
from pydantic import BaseModel
from moz_sql_parser import parse

from raster_analysis.data_environment import DataEnvironment
from raster_analysis.layer import Layer, Grid
from raster_analysis.data_lake import LAYERS
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
    layer: Layer
    value: Any

    def apply_filter(self, window: ndarray) -> ndarray:
        return eval(f"window {self.operator.value} self.value")


class AggregateFunction(str, Enum):
    sum = "sum"
    avg = "avg"
    count_ = "count"


class Aggregate(BaseModel):
    function: AggregateFunction
    layer: Layer


class Query(BaseModel):
    base: Layer
    selectors: List[Layer]
    filters: List[Filter] = []
    groups: List[Layer] = []
    aggregates: List[Aggregate] = []
    data_environment: DataEnvironment

    def get_real_layers(self) -> List[Layer]:
        layers = self.get_layers()
        return [
            layer for layer in layers if layer.layer not in SpecialSelectors.__members__
        ]

    def get_layers(self) -> List[Layer]:
        layers = [selector for selector in self.selectors]
        layers += [filter.layer for filter in self.filters]
        layers += [
            agg.layer
            for agg in self.aggregates
            if agg.function != AggregateFunction.count_
        ]
        layers += [group for group in self.groups]

        if self.has_layer(self.base.layer):  # self.base.alias in LAYERS:
            layers.append(self.base)

        return list(dict.fromkeys(layers))

    def get_result_layers(self) -> List[Layer]:
        layers = [selector for selector in self.selectors]
        layers += [aggregate.layer for aggregate in self.aggregates]
        layers += [group for group in self.groups]

        return list(dict.fromkeys(layers))

    def get_group_columns(self) -> List[str]:
        return [group.layer for group in self.groups]

    def get_minimum_grid(self) -> Grid:
        layers = self.get_layers()
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

        base = data_environment.get_layer([parsed["from"]])
        for selector in Query._ensure_list(parsed["select"]):
            if isinstance(selector["value"], dict):
                func, layer_name = Query._get_first_key_value(selector["value"])
                layer = data_environment.get_layer(layer_name)
                aggregate = Aggregate(function=func, layer=layer)
                aggregates.append(aggregate)
            elif isinstance(selector["value"], str):
                selectors.append(data_environment.get_layer(selector["value"]))

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
                if layer.encoder:
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
                groups.append(data_environment.get_layer(group["value"]))

        return Query(
            base=base,
            selectors=selectors,
            filters=where,
            groups=groups,
            aggregates=aggregates,
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
