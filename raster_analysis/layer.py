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


