from raster_analysis import geoprocessing
from shapely.geometry import shape


def lambda_handler(event, context):
    try:
        raster_ids = event["raster_ids"]
        analysis = event["analysis"]
        geometry = shape(event["geometry"])
        threshold = event["threshold"]
    except KeyError:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": {"message": "Bad input parameters."},
        }

    try:
        body = geoprocessing.analysis(
            geometry, *raster_ids, threshold=threshold, analysis=analysis
        )

        result = {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": body,
        }

    except Exception as e:
        result = {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": {"message": e},
        }

    return result
