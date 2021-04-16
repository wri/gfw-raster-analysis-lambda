import json
from enum import Enum
from typing import Optional, Dict, Any, List, Union

from pydantic import BaseModel, parse_obj_as

from raster_analysis.globals import BasePolygon


class RealLayer(BaseModel):
    source_uri: str
    tile_scheme: TileScheme
    grid: GridName
    name: str
    pixel_encoding: Dict[int, Any] = {}


class DerivedLayer(BaseModel):
    source_layer: str
    name: str
    derivation_expression: str


class DataEnvironment:
    def __init__(self, layers: List[Dict[str, Any]]):
        self.layers = parse_obj_as(layers, List[Union[RealLayer, DerivedLayer])

    def get_layer(self, name: str) -> RealLayer:
        for layer in self.layers:
            if layer.name == name:
                return layer

        raise KeyError(f"Could not find layer with name {name} in data environment {json.dumps(self.layers)}")

    def get_layer_grid(self, name: str) -> Grid:
        layer = self.get_layer(name)
        return Grid.get_grid(layer.grid)

    def has_layer(self, name: str) -> bool:
        try:
            self.get_layer(name)
            return True
        except ValueError:
            return False

    def get_real_layers(self):
        pass


    def get_source_uri(self, name: str, geometry: BasePolygon) -> str:
        layer = self.get_layer(name)

        if layer.source_uri:
            tile_id = layer.grid.get_tile_id(geometry, layer.tile_scheme)
            return layer.source_uri.format(tile_id)
        elif layer.source_layer:
            return self.get_source_uri(layer.source_layer, geometry)
        else:
            raise ValueError(f"Cannot get source URI for layer {name}")

    def decode_layer(self, name: str):
        layer = self.get_layer(name)
        if layer.pixel_encoding:


TEST = [
    {
        "source_uri": "s3://gfw-data-lake-staging/umd_tree_cover_loss/v1.8/raster/epsg-4326/10/40000/year/geotiff/{tile_id}.tif",
        "tile_scheme": "nw",
        "grid": "10/40000",
        "name": "umd_tree_cover_loss__year",
        "pixel_encoding": {
            1: 2001,
            2: 2002,
            3: 2003,
            4: 2004,
            5: 2005,
        }
    },
    {
        "source_uri": "s3://gfw2-data/forest_change/umd_landsat_alerts/prod/analysis/{tile_id}.tif",
        "tile_scheme": "nwse",
        "grid": "10/40000",
        "type": "date_conf",
        "name": "umd_glad_landsat_alerts__date_conf",
    },
    {
        "source_layer": "umd_glad_landsat_alerts__date_conf",
        "name": "umd_glad_landsat_alerts__date",
        "derivation_expression": "(A % 10000).astype('timedelta64[D]') + datetime64('2015-01-01')",
    },
    {
        "source_layer": "umd_glad_landsat_alerts__date_conf",
        "name": "umd_glad_landsat_alerts__date",
        "derivation_expression": "floor(A / 10000)",
        "encoding": {
            2: "",
            3: "high"
        }
    },
]
