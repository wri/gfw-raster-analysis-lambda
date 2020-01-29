import logging

logger = logging.getLogger(__name__)

BASE_URL = "/vsis3/test-analysis-data/{raster_id}.vrt"


def get_raster_url(raster_id):
    return BASE_URL.format(raster_id=raster_id)
