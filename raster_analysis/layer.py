from typing import Any, Dict, List, Callable

from pydantic import BaseModel
from pandas import Series


class Layer(BaseModel):
    layer: str
    version: str
    decoder: Callable[[Series], Dict[str, Series]] = None
    encoder: Callable[[Any], List[Any]] = None
    is_area_density: bool = False
    is_conf_encoded: bool = False

    def __hash__(self):
        return hash(self.layer)

    @staticmethod
    def from_encoding(layer: str, version: str, encoding: Dict[Any, Any], is_area_density: bool = False, is_conf_encoded: bool = False):
        decoder = (lambda s: {layer: s.map(encoding)})
        encoder = (lambda val: [enc_val for enc_val, dec_val in encoding.items() if val == dec_val])
        return Layer(
            layer=layer,
            version=version,
            encoder=encoder,
            decoder=decoder,
            is_area_density=is_area_density,
            is_conf_encoded=is_conf_encoded
        )

    # def __init__(
    #         self,
    #         layer: str,
    #         version: str,
    #         encoding: Dict[Any, Any] = {},
    #         decoder: Callable[[Series], Dict[str, Series]] = None,
    #         encoder: Callable[[Any], List[Any]] = None,
    #         is_area_density: bool = False,
    #         is_conf_encoded: bool = False,
    # ):
    #     self.layer: str = layer
    #     self.version: str = version
    #     self.is_area_density: bool = is_area_density
    #     self.is_conf_encoded: bool = is_conf_encoded
    #     if encoding:
    #         self.decoder = (lambda s: {layer: s.map(encoding)})
    #         self.encoder = (lambda val: [dec_val for dec_val in self.encoding.values() if self.value == dec_val])
    #     else:
    #         self.decoder: Callable[[Series], Dict[str, Series]] = decoder
    #         self.encoder: Callable[[Any], List[Any]] = encoder
    #



