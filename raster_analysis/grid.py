import logging

logger = logging.getLogger(__name__)

BASE_URL = "/vsis3/gfw-data-lake-dev/{raster_id_no_unit}/latest/raster/10x10/{raster_id}/{raster_id}.vrt"


def get_raster_url(raster_id):
    raster_id_parts = raster_id.split("__")
    raster_id_no_unit = (
        raster_id_parts[1] if raster_id_parts[0] == "is" else raster_id_parts[0]
    )
    return BASE_URL.format(raster_id=raster_id, raster_id_no_unit=raster_id_no_unit)
