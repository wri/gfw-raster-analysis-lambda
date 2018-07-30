from flask import jsonify


def api_error(msg):
    print msg

    return jsonify({'error': msg}), 400


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

    return jsonify(serialized), 200

