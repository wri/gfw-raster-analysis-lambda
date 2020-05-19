import logging
import os

logger = logging.getLogger(__name__)


def get_raster_url(layer: str) -> str:
    """
    Maps layer name input to a raster URI in the data lake
    :param layer: Either of format <layer name>__<unit> or <unit>__<layer>
    :return: A GDAL (vsis3) URI to the corresponding VRT for the layer in the data lake
    """
    parts = layer.split("__")

    if parts != 2:
        raise Exception(
            f"Layer name `{layer}` is invalid, should consist of layer name and unit separated by `__`"
        )

    if parts[0] == "is":
        unit, layer = parts
    else:
        layer, unit = parts

    return f"/vsis3/{os.environ['S3_BUCKET_DATA_LAKE']}/{layer}/raster/epsg-4326/10/40000/{unit}/gdal-geotiff/all.vrt"
