# flake8: noqa
import json
from collections import Iterable, defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast

from numpy import (
    ceil,
    datetime64,
    float,
    float32,
    float64,
    floor,
    timedelta64,
    uint,
    uint16,
    uint32,
    uint64,
)
from pandas import Series
from pydantic import BaseModel, parse_obj_as

from raster_analysis.globals import BasePolygon
from raster_analysis.grid import Grid, GridName, TileScheme


class SourceLayer(BaseModel):
    source_uri: str
    name: str
    tile_scheme: TileScheme = TileScheme.nw
    grid: GridName = GridName.ten_by_forty_thousand
    pixel_encoding: Dict[Any, Any] = {}
    decode_expression: str = ""
    encode_expression: str = ""


class DerivedLayer(BaseModel):
    source_layer: str
    name: str
    derivation_expression: str
    pixel_encoding: Dict[Any, Any] = {}
    decode_expression: str = ""
    encode_expression: str = ""


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

    def get_layer(self, name: str) -> Layer:
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
        return [self.get_layer(layer_name) for layer_name in layer_names]

    def get_source_layers(self, layer_names: List[str]) -> List[SourceLayer]:
        layers = self.get_layers(layer_names)

        source_layers = []
        for layer in layers:
            if isinstance(layer, SourceLayer):
                layer = cast(SourceLayer, layer)
                source_layers.append(layer)
            elif isinstance(layer, DerivedLayer):
                layer = cast(DerivedLayer, layer)
                source_layer = self.get_layer(layer.source_layer)
                source_layers.append(source_layer)

        return source_layers

    def get_derived_layers(self, layer_names: List[str]) -> List[DerivedLayer]:
        layers = self.get_layers(layer_names)
        return [layer for layer in layers if isinstance(layer, DerivedLayer)]

    def encode_layer(self, name: str, val: Any) -> List[Any]:
        layer = self.get_layer(name)
        if layer.pixel_encoding:
            encoded_vals = [
                encoded_val
                for encoded_val, decoded_val in layer.pixel_encoding.items()
                if val == decoded_val
            ]

            if not encoded_vals:
                raise ValueError(
                    f"Value {val} not in pixel encoding {layer.pixel_encoding}"
                )
            return encoded_vals
        elif layer.encode_expression:
            A = val
            result = eval(layer.encode_expression)
            return list(result) if isinstance(result, Iterable) else [result]
        else:
            # if no pixel_encoding, encoded value is just the same as decoded value
            return [val]

    def decode_layer(self, name: str, values: Series) -> Series:
        layer = self.get_layer(name)

        if isinstance(layer, ReservedLayer):
            return values

        pixel_encoding = self.get_pixel_encoding(name)
        if pixel_encoding:
            return values.map(pixel_encoding)
        elif layer.decode_expression:
            A = values
            return eval(layer.decode_expression)

    def get_source_uri(self, name: str, tile: BasePolygon):
        layer = self.get_layer(name)

        if layer.tile_scheme:
            grid = Grid.get_grid(layer.grid)
            tile_id = grid.get_tile_id(tile, layer.tile_scheme)
            return layer.source_uri.format(tile_id=tile_id)
        else:
            return layer.source_uri

    def get_pixel_encoding(self, name: str):
        layer = self.get_layer(name)

        # default value is implemented in Python as a defaultdict
        if isinstance(layer, SourceLayer):
            if "_" in layer.pixel_encoding:
                encoding = deepcopy(layer.pixel_encoding)
                default_val = encoding["_"]
                del encoding["_"]

                return defaultdict(lambda: default_val, encoding)

            return layer.pixel_encoding
        else:
            return {}

    def has_default_value(self, name):
        """A layer is defined as having a default value if it's encoding
        contains the default value symbol '_', or a mapping for 0.

        Otherwise, 0 is considered NoData and will be filtered out.
        """
        layer = self.get_layer(name)
        return "_" in layer.pixel_encoding or 0 in layer.pixel_encoding
