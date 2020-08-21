import numpy as np
from aws_xray_sdk.core import xray_recorder

from typing import Dict, List
from numpy import ndarray

DataFrame = Dict[str, ndarray]


@xray_recorder.capture("Get Linear Index")
def get_linear_index(
    cols: List[ndarray], dims: List[int], mask: ndarray = None
) -> ndarray:
    linear_index = np.ravel_multi_index(cols, dims).astype(np.uint32)
    if mask is not None:
        linear_index = np.compress(np.ravel(mask), linear_index)

    return linear_index
