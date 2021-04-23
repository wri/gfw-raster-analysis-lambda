import json
from typing import Optional, Dict, Any, List, Union, Callable, Type, cast

from pydantic import BaseModel, parse_obj_as

from raster_analysis.grid import GridName, TileScheme, Grid


class SourceLayer(BaseModel):
    source_uri: str
    tile_scheme: TileScheme
    grid: GridName
    name: str
    pixel_encoding: Dict[int, Any] = {}


class DerivedLayer(BaseModel):
    source_layer: str
    name: str
    derivation_expression: str


class ReservedLayer(BaseModel):
    name: str


Layer = Union[SourceLayer, DerivedLayer, ReservedLayer]


RESERVED_LAYERS = [
    ReservedLayer(name="area__ha"),
    ReservedLayer(name="latitude"),
    ReservedLayer(name="longitude"),
    ReservedLayer(name="alert__count"),
]


class DataEnvironment(BaseModel):
    layers: List[Layer]

    def get_layer(self, name: str) -> SourceLayer:
        for layer in self.layers + RESERVED_LAYERS:
            if layer.name == name:
                return layer

        raise KeyError(
            f"Could not find layer with name {name} in data environment {json.dumps(self.dict())}"
        )

    def get_layer_grid(self, name: str) -> Grid:
        layer = self.get_layer(name)
        return Grid.get_grid(layer.grid)

    def has_layer(self, name: str) -> bool:
        try:
            self.get_layer(name)
            return True
        except ValueError:
            return False

    def get_layers(self, layer_names: List[str]) -> List[Layer]:
        layers = []
        for layer_name in layer_names:
            layer = self.get_layer(layer_name)
            layers.append(layer)

        return list(dict.fromkeys(layers))

    def get_source_layers(self, layer_names: List[str]) -> List[SourceLayer]:
        layers = self.get_layers(layer_names)

        source_layers = []
        for layer in layers:
            if isinstance(layer, SourceLayer):
                layer = cast(SourceLayer, layer)
                source_layers.append(layer)
            elif isinstance(layer, SourceLayer):
                layer = cast(DerivedLayer, layer)
                source_layer = self.get_layer(layer.source_layer)
                source_layers.append(source_layer)

        return self.get_layers(layer_names, SourceLayer)

    def get_derived_layers(self, layer_names: List[str]) -> List[DerivedLayer]:
        layers = self.get_layers(layer_names)
        return [layer for layer in layers if isinstance(layer, DerivedLayer)]
