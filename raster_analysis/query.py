from typing import List, Any, Set
from enum import Enum

from pydantic import BaseModel

from raster_analysis.globals import DATA_LAKE_LAYER_MANAGER

AREA_DENSITY_TYPE = "ha-1"
EMISSIONS_LAYER = "whrc_aboveground_co2_emissions__Mg"


class LayerInfo(BaseModel):
    name: str
    type: str
    is_area_density: bool = False
    is_emissions: bool = False

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
        elif layer == EMISSIONS_LAYER:
            self.is_emissions = True


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
