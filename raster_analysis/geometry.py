import sys
from typing import Any, Dict, Optional

import geobuf
from shapely.geometry import Polygon, mapping, shape

from raster_analysis.exceptions import InvalidGeometryException
from raster_analysis.globals import LAMBDA_ASYNC_PAYLOAD_LIMIT_BYTES, BasePolygon


class GeometryTile:
    def __init__(
        self,
        source_geom: str,
        tile_geojson: Optional[Dict[str, Any]] = None,
        is_encoded: bool = False,
    ):
        if is_encoded:
            full_geom = decode_geometry(source_geom)
        else:
            full_geom = shape(source_geom).buffer(0)

        self.geom: BasePolygon = full_geom
        self.tile: Optional[Polygon] = None

        if tile_geojson:
            tile = shape(tile_geojson)
            self.tile = tile

            geom_tile = full_geom.intersection(tile)

            if not geom_tile.is_valid:
                geom_tile = geom_tile.buffer(0)

                if not geom_tile.is_valid:
                    raise InvalidGeometryException(
                        f"Could not create valid tile from geom {full_geom.wkt} and tile {tile.wkt}"
                    )

            self.geom = geom_tile


def encode_geometry(geom: BasePolygon) -> str:
    """Encode geometry into a compressed string."""
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


def decode_geometry(geom: str) -> BasePolygon:
    """Decode geometry from compressed string."""
    return shape(geobuf.decode(bytes.fromhex(geom))).buffer(0)
