# flake8: noqa
import json
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from typing import Any, Dict, List, Optional, Union, cast

from numpy import (
    ceil,
    datetime64,
    float32,
    float64,
    floor,
    isnan,
    nan,
    timedelta64,
    uint,
    uint8,
    uint16,
    uint32,
    uint64,
)
from pandas import Series
from pydantic import BaseModel, validator

from raster_analysis.exceptions import QueryParseException
from raster_analysis.globals import LOGGER, BasePolygon, Numeric
from raster_analysis.grid import Grid, GridName, TileScheme


class RasterTableRow(BaseModel):
    value: int
    meaning: Any


class RasterTable(BaseModel):
    rows: List[RasterTableRow]
    default_meaning: Any = None


class BaseLayer(BaseModel):
    name: str
    no_data: Optional[Numeric] = 0

    @validator("no_data")
    def parse_no_data(cls, v):
        if v == "nan":
            return nan
        return v


class EncodedLayer(BaseLayer):
    raster_table: Optional[RasterTable] = None
    decode_expression: str = ""
    encode_expression: str = ""


class SourceLayer(EncodedLayer):
    source_uri: str
    tile_scheme: TileScheme = TileScheme.nw
    grid: GridName = GridName.ten_by_forty_thousand


class DerivedLayer(EncodedLayer):
    source_layer: str
    calc: str


class ReservedLayer(BaseLayer):
    pass


Layer = Union[SourceLayer, DerivedLayer, ReservedLayer]


RESERVED_LAYERS = [
    ReservedLayer(name="area__ha"),
    ReservedLayer(name="latitude"),
    ReservedLayer(name="longitude"),
]


class DataEnvironment(BaseModel):
    layers: List[Layer]

    def get_layer(self, name: str) -> Layer:
        for layer in self.layers + RESERVED_LAYERS:
            if layer.name == name:
                return layer

        if name.endswith("__ha"):
            for layer in self.layers:
                base_name = name.replace("__ha", "")
                if layer.name.endswith(f"__{base_name}") or layer.name.startswith(
                    f"{base_name}__"
                ):
                    return DerivedLayer(
                        name=name,
                        source_layer=layer.name,
                        calc="np.where(A > 0, area, 0)",
                    )

        LOGGER.error(
            f"Could not find layer with name {name} in data environment {json.dumps(self.dict())}"
        )
        raise QueryParseException(f"Layer {name} is invalid")

    def get_layer_grid(self, name: str) -> Grid:
        layer = self.get_layer(name)
        if isinstance(layer, SourceLayer):
            return Grid.get_grid(layer.grid)
        else:
            raise Exception("Tried to get grid of a non-SourceLayer!")

    def has_layer(self, name: str) -> bool:
        try:
            self.get_layer(name)
            return True
        except QueryParseException:
            return False

    def get_layers(self, layer_names: List[str]) -> List[Layer]:
        return [self.get_layer(layer_name) for layer_name in layer_names]

    def get_source_layers(self, layer_names: List[str]) -> List[SourceLayer]:
        layers = self.get_layers(layer_names)

        source_layers: List[SourceLayer] = []
        for layer in layers:
            if isinstance(layer, SourceLayer):
                source_layers.append(layer)
            elif isinstance(layer, DerivedLayer):
                source_layer: SourceLayer = self.get_layer(layer.source_layer)
                source_layers.append(source_layer)

        return source_layers

    def get_derived_layers(self, layer_names: List[str]) -> List[DerivedLayer]:
        layers = self.get_layers(layer_names)
        return [layer for layer in layers if isinstance(layer, DerivedLayer)]

    def encode_layer(self, name: str, val: Any) -> List[Any]:
        layer = self.get_layer(name)
        if isinstance(layer, EncodedLayer):
            encoding = self.get_pixel_encoding(name)
            if encoding:
                encoded_vals = [
                    encoded_val
                    for encoded_val, decoded_val in encoding.items()
                    if val == decoded_val
                ]

                # if encoding has default value, encode to NoData
                if isinstance(encoding, defaultdict) and val == encoding[0]:
                    encoded_vals.append(0)

                if not encoded_vals:
                    raise ValueError(f"Value {val} not in pixel encoding {encoding}")
                return encoded_vals
            elif layer.encode_expression:
                A = val
                result = eval(layer.encode_expression)
                return list(result) if isinstance(result, Iterable) else [result]

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
        else:
            return values

    def get_source_uri(self, name: str, tile: BasePolygon):
        layer = self.get_layer(name)

        if isinstance(layer, SourceLayer):
            grid = Grid.get_grid(GridName(layer.grid))
            tile_id = grid.get_tile_id(tile, layer.tile_scheme)
            source_uri = layer.source_uri.format(tile_id=tile_id)

            if "s3" in source_uri:
                source_uri = source_uri.replace("s3://", "/vsis3/")

            return source_uri
        else:
            raise Exception("Tried to call get_source_uri on a non-SourceLayer!")

    def get_pixel_encoding(self, name: str) -> Dict[int, Any]:
        layer = self.get_layer(name)

        if isinstance(layer, EncodedLayer) and layer.raster_table:
            raster_table = layer.raster_table
            cast(RasterTable, raster_table)

            encoding = {row.value: row.meaning for row in raster_table.rows}

            # default value is implemented in Python as a defaultdict
            if layer.raster_table.default_meaning:
                encoding = defaultdict(lambda: raster_table.default_meaning, encoding)

            return encoding
        else:
            return {}

    def has_default_value(self, name):
        """A layer is defined as having a default value if its encoding
        contains the default value symbol '_', or a mapping for 0.

        Otherwise, 0 is considered NoData and will be filtered out.
        """
        encoding = self.get_pixel_encoding(name)
        return isinstance(encoding, defaultdict) or 0 in encoding

    def dict(self, *args, **kwargs):
        layers = super().dict()["layers"]

        # nan values must be serialized as strings since they're not JSON-compliant
        for layer in layers:
            if layer["no_data"] is not None and isnan(layer["no_data"]):
                layer["no_data"] = "nan"

        return layers
