from raster_analysis import geoprocessing
from shapely.geometry import shape

import sys
import json

BASE_URL = "/vsis3/gfw-files/2018_update/{raster_id}/{tile_id}.tif"


def lambda_handler(event, context):
    raster_ids = event["raster_ids"]
    analysis = event["analysis"]
    geometry = shape(event["geometry"])
    threshold = event["threshold"]

    try:
        result = geoprocessing.analysis(
            geometry, *raster_ids, threshold=threshold, analysis=analysis
        )
    except Exception as e:
        result = {"status": 500,
                  "message": e}

    return result


if __name__ == "__main__":
    #     #"{\"raster_ids\":[\"loss\", \"tcd_2000\", \"tcd_2010\", \"wdpa\"], \"analysis\":\"area_sum\", \"threshold\":30, \"geometry\":{\"type\":\"Polygon\",\"coordinates\":[[[9.0,4.1],[9.1,4.1],[9.1,4.2],[9.0,4.2],[9.0,4.1]]]}}"
    print(lambda_handler(json.loads(sys.argv[1]), None))
