from raster_analysis import geoprocessing
from raster_analysis.geoprocessing import Filter
from shapely.geometry import shape

import sys
import json

BASE_URL = "/vsis3/gfw-files/2018_update/{raster_id}/{tile_id}.tif"


def serialize(func):
    def wrapper(*args, **kwargs):
        try:
            body = func(*args, **kwargs)
            result = {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": body,
            }
        except ValueError as e:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": {"message": e},
            }
        except Exception as e:
            result = {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": {"message": e},
            }

        return result

    return wrapper


@serialize
def lambda_handler(event, context):
    missing_params = [
        param not in event for param in ["raster_ids", "analysis", "geometry"]
    ]
    if len(missing_params) != 0:
        raise ValueError("Missing parameters: " + ", ".join(missing_params))

    raster_ids = event["raster_ids"]
    analysis = event["analysis"]
    geometry = shape(event["geometry"])

    if analysis not in ["area", "sum", "count"]:
        raise ValueError("Unknown analysis: " + analysis)

    filters = [Filter(**f) for f in event["filters"]] if "filters" in event else []

    return geoprocessing.analysis(
        geometry, *raster_ids, filters=filters, analysis=analysis
    )


if __name__ == "__main__":
    #     #"{\"raster_ids\":[\"loss\", \"tcd_2000\", \"tcd_2010\", \"wdpa\"], \"analysis\":\"area_sum\", \"threshold\":30, \"geometry\":{\"type\":\"Polygon\",\"coordinates\":[[[9.0,4.1],[9.1,4.1],[9.1,4.2],[9.0,4.2],[9.0,4.1]]]}}"
    print(lambda_handler(json.loads(sys.argv[1]), None))
