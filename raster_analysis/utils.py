from typing import List
import sys

import numpy as np
from numpy import ndarray
from aws_xray_sdk.core import xray_recorder
from shapely.geometry import mapping, shape
import geobuf

from raster_analysis.globals import LAMBDA_ASYNC_PAYLOAD_LIMIT_BYTES, BasePolygon


@xray_recorder.capture("Get Linear Index")
def get_linear_index(
    cols: List[ndarray], dims: List[int], mask: ndarray = None
) -> ndarray:
    linear_index = np.ravel_multi_index(cols, dims).astype(np.uint32)
    if mask is not None:
        linear_index = np.compress(np.ravel(mask), linear_index)

    return linear_index


@xray_recorder.capture("Encode Geometry")
def encode_geometry(geom: BasePolygon) -> str:
    """
    Encode geometry into a compressed string
    """
    encoded_geom = geobuf.encode(mapping(geom)).hex()

    # if the geometry is so complex is still goes over the limit, incrementally attempting to simplify it
    if sys.getsizeof(encoded_geom) > LAMBDA_ASYNC_PAYLOAD_LIMIT_BYTES:
        encoded_geom = geobuf.encode(
            mapping(geom.simplify(0.005, preserve_topology=False))
        ).hex()

    if sys.getsizeof(encoded_geom) > LAMBDA_ASYNC_PAYLOAD_LIMIT_BYTES:
        encoded_geom = geobuf.encode(
            mapping(geom.simplify(0.01, preserve_topology=False))
        ).hex()

    return encoded_geom


@xray_recorder.capture("Decode Geometry")
def decode_geometry(geom: str) -> BasePolygon:
    """
    Decode geometry from compressed string
    """
    return shape(geobuf.decode(bytes.fromhex(geom))).buffer(0)
