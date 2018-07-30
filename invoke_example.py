import boto3
import json


session = boto3.Session(profile_name='gfwpro')
client = session.client('lambda', region_name='us-east-1')

aoi = {"features":[{"properties":{},"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[140.2137,-6.3999],[140.3078,-6.3685],[140.3249,-6.3924],[140.3249,-6.4606],[140.3064,-6.49],[140.2274,-6.4838],[140.2068,-6.434],[140.2137,-6.3999]]]}}],"crs":{},"type":"FeatureCollection"}

# then build an event to kick off the process
event = {'queryStringParameters': {'aggregate_values': True, 'aggregate_by': 'day'}, 'body': {'geojson': aoi}}


# invoke our bulk upload function, which will download the
# fire_csv above, split it into 1x1 tiles, then kick off
# per-tile lambda functions to store this data
resp = client.invoke(
        FunctionName='geoproc-raster-glad',
        InvocationType='RequestResponse',
        Payload=json.dumps(event))


print resp['Payload'].read()
