import json

from utilities import lulc_util


def http_response(response):

    # print json.dumps(response, indent=4, sort_keys=True)

    return {
        'statusCode': 200,
        'headers': {'Access-Control-Allow-Origin': '*'},
        'body': json.dumps(response)
            }


def api_error(msg):
    print msg

    return {
        'statusCode': 400,
        'headers': {'Access-Control-Allow-Origin': '*'},
        'body': json.dumps({'error': msg})
            }


def serialize_glad(hist, area_ha, agg_by, period): 
    serialized = {
    "data": {
        "aggregate_by": agg_by,
        "aggregate_values": True,
        "attributes": {
            "area_ha": area_ha,
            "downloadUrls": None,
            "value": hist},
        "period": period,
        "type": "glad-alerts"}
    }

    return http_response(serialized)

