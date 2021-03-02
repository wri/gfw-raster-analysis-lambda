from typing import List, Any, Set, Dict
from enum import Enum

from pydantic import BaseModel
from moz_sql_parser import parse

from raster_analysis.exceptions import QueryParseException
from raster_analysis.globals import DATA_LAKE_LAYER_MANAGER

AREA_DENSITY_TYPE = "ha-1"


class LayerInfo:
    def __init__(self, layer: str):
        parts = layer.split("__")

        if len(parts) != 2:
            raise ValueError(
                f"Layer name `{layer}` is invalid, should consist of layer name and unit separated by `__`"
            )

        if parts[0] == "is":
            self.type, self.name = parts
        else:
            self.name, self.type = parts

        if AREA_DENSITY_TYPE in self.type:
            self.is_area_density = True


class SpecialSelectors(str, Enum):
    latitude = "latitude"
    longitude = "longitude"
    area = "area__ha"


class Operator(str, Enum):
    gt = ">"
    lt = "<"
    gte = ">="
    lte = "<="
    eq = "=="
    neq = "!="


class Filter(BaseModel):
    operator: Operator
    layer: LayerInfo
    value: Any

    def apply_filter(self, window):
        DATA_LAKE_LAYER_MANAGER.get_layer_value()
        return eval(f"window {self.operator.value} self.value")


class AggregateFunction(str, Enum):
    sum = "sum"
    avg = "avg"


class Aggregate(BaseModel):
    function: AggregateFunction
    layer: LayerInfo


class Query(BaseModel):
    selectors: List[LayerInfo] = []
    filters: List[Filter] = []
    groups: List[LayerInfo] = []
    aggregates: List[Aggregate] = []

    def get_layers(self) -> Set[LayerInfo]:
        layers = [selector.layer for selector in self.selectors]
        layers += [filter.layer for filter in self.filters]
        layers += [aggregate.layer for aggregate in self.aggregates]

        return set(layers)


def parse_query(raw_query: str) -> Query:
    parsed = parse(raw_query)
    query = Query()

    if "select" not in parsed:
        raise QueryParseException("Query be SELECT statement")

    for selector in _ensure_list(parsed["select"]):
        if isinstance(selector["value"], dict):
            func, layer = _get_first_key_value(selector["value"])
            aggregate = Aggregate(function=func, layer=LayerInfo(layer))
            query.aggregates.append(aggregate)
        elif isinstance(selector["value"], str):
            query.selectors.append(LayerInfo(selector["value"]))

    if "where" in parsed:
        for filter in _ensure_list(parsed["where"]):
            op, (layer, value) = _get_first_key_value(filter)
            layer = LayerInfo(layer)
            encoded_values = DATA_LAKE_LAYER_MANAGER.layers[layer].encode(value)
            for encoded_value in encoded_values:
                query.filters.append(Filter(operator=op, layer=LayerInfo(layer), value=encoded_value))

    if "groupby" in parsed:
        for group in _ensure_list(parsed["groupby"]):
            query.groups.append(LayerInfo[group["value"]])


# the SQL parser sometimes outputs a list or single value depending on input,
# Just a helper to make it consistent
def _ensure_list(a):
    return a if type(a) is list else [a]


# certain things are parsed single key value pairs, so just get the first key value pair
def _get_first_key_value(d: Dict[str, str]):
    for key, val in d.items():
        return (key, val)
