import os
import boto3
import json
import uuid


client = boto3.client('lambda')

# load geojson test points
# source: q
cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cwd, 'idn_points.geojson')) as thefile:
    data = json.load(thefile)

for feature in data['features']:

    lon, lat = feature['geometry']['coordinates']

    # build an out dir that includes lat and lon and random GUID
    # will probably change this in the future, but nice right now
    # to have lat/lon in there for debugging
    out_dir = '{}_{}_{}'.format(round(lat, 5), round(lon, 5), str(uuid.uuid4()))

    event = {
    'queryStringParameters': {
        'out_dir': out_dir,
        'lat': lat,
        'lon': lon
        }
    }

    client.invoke(
        FunctionName='palm-risk-poc-dev-mill',
        InvocationType='Event',
        Payload=json.dumps(event)
        )
