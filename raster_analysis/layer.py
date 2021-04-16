from typing import Any, Dict, List, Callable, Optional
from collections import defaultdict

from pydantic import BaseModel
from pandas import Series


class Grid(BaseModel):
    degrees: int
    pixels: int
    tile_degrees: float

    def get_pixel_width(self) -> float:
        return self.degrees / self.pixels

    def get_tile_width(self) -> int:
        return round((self.tile_degrees / self.degrees) * self.pixels)


# class Layer(BaseModel):
#     layer: str
#     version: str
#     grid: Grid = Grid(degrees=10, pixels=40000, tile_degrees=1.25)
#     alias: Optional[str] = None
#     decoder: Optional[Callable[[Any, Series], Dict[str, Series]]] = None
#     encoder: Optional[Callable[[Any], List[Any]]] = None
#     has_default_value: bool = False
#     is_area_density: bool = False
#
#     def __hash__(self):
#         return hash(self.layer)
#
#     @staticmethod
#     def from_encoding(
#         layer: str,
#         version: str,
#         encoding: Dict[Any, Any],
#         grid: Grid = Grid(degrees=10, pixels=40000, tile_degrees=1.25),
#         alias: Optional[str] = None,
#         is_area_density: bool = False,
#     ):
#         has_default_value = 0 in encoding or isinstance(encoding, defaultdict)
#
#         def decode(lyr, s):
#             return {(alias if alias else layer): s.map(encoding)}
#
#         def encode(val):
#             return [enc_val for enc_val, dec_val in encoding.items() if val == dec_val]
#
#         decoder = decode
#         encoder = encode
#         return Layer(
#             layer=layer,
#             version=version,
#             encoder=encoder,
#             decoder=decoder,
#             alias=alias,
#             has_default_value=has_default_value,
#             is_area_density=is_area_density,
#             grid=grid,
#         )
#
#     @staticmethod
#     def boolean(*args, **kwargs):
#         kwargs["encoding"] = {0: "false", 1: "true"}
#         return Layer.from_encoding(*args, **kwargs)
