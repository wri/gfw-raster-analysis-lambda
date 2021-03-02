from typing import Any, Dict

from shapely.geometry import shape, Polygon

from raster_analysis.exceptions import InvalidGeometryException
from raster_analysis.utils import decode_geometry
from raster_analysis.globals import BasePolygon


class GeometryTile:
    def __init__(self, source_geom: Any, tile_geojson: Dict[str, Any] = None, is_encoded: bool = False):
        if is_encoded:
            full_geom = decode_geometry(source_geom)
        else:
            full_geom = shape(source_geom)

        self.geom: BasePolygon = full_geom
        self.tile: Polygon = None

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

            if geom_tile.is_empty:
                self.geom = {}

            self.geom: BasePolygon = geom_tile

