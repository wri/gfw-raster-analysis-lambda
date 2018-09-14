import json
import uuid
import csv
import io

import boto3
from flask import jsonify

s3 = boto3.resource('s3')


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


def write_to_s3(rows, out_format):

    guid = str(uuid.uuid4())
    output_key = 'data/glad-download/{}.{}'.format(guid, out_format)

    output_bucket = 'palm-risk-poc'
    s3_output = s3.Object(output_bucket, output_key)

    # create out CSV text string
    if out_format == 'csv':
        out_csv = io.BytesIO()
        writer = csv.writer(out_csv)

        writer.writerow(['longitude', 'latitude', 'year', 'julian_day', 'confidence'])

        for row in rows:
            writer.writerow(row)

        s3_output.put(Body=out_csv.getvalue())
    else:
        s3_output.put(Body=json.dumps({'data': rows}))

    output_url = 'http://{}.s3.amazonaws.com/{}'.format(output_bucket, output_key)

    return output_url

