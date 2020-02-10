import logging
import os

logger = logging.getLogger(__name__)
DATA_LAKE_BUCKET = os.environ["S3_BUCKET_DATA_LAKE"]
BASE_URL = f"/vsis3/DATA_LAKE_BUCKET"


def get_raster_url(raster_id):
    return f"{BASE_URL}/{raster_id}.vrt"
