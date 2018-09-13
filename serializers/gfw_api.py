import json

from flask import jsonify


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


def stream_download(rows, out_format):

    # CSV is easy - header, then row by row
    if out_format == 'csv':
        yield 'longitude,latitude,year,julian_day,confidence\n'

        for row in rows:
            yield row

    # JSON is harder - need to join each row with a comma,
    # but can't have a comma after the final row
    # from: https://blog.al4.co.nz/2016/01/streaming-json-with-flask/
    else:
        yield '{"data": ['

        prev_row = next(rows, None)

        for row in rows:
            yield json.dumps(row) + ', '
            prev_row = row

        if prev_row:
            yield json.dumps(prev_row) + ']}'
        else:
            yield ']}'

