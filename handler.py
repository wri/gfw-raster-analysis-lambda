import json
import boto3
import uuid

from shapely.geometry import shape
from geop import geo_utils, geoprocessing

s3 = boto3.resource('s3')
bucket_uri = 'palm-risk-poc'
client = boto3.client('lambda')


# source https://stackoverflow.com/questions/34294693
# return a quick response + then kick off async functions
def receiver(event, context):
    # define mill guid and pass along to mil event
    mill_guid = str(uuid.uuid4())
    event['out_dir'] = mill_guid

    client.invoke(
        FunctionName='palm-risk-poc-dev-mill',
        InvocationType='Event',
        Payload=json.dumps(event))

    return {
        'statusCode': 200,
        'headers': {'Access-Control-Allow-Origin': '*'},
        'body': json.dumps({"result": "starting mill processing, check "
                                      "s3://{}/output/{}/ for results".format(bucket_uri, mill_guid)
                            })
            }


# buffer a point, then kick off parallel lambda
# functions for risk analyses
def mill(event, context):

    try:
        lat = float(event['queryStringParameters']['lat'])
        lon = float(event['queryStringParameters']['lon'])
        event['out_dir'] = event['queryStringParameters']['out_dir']

    except TypeError, KeyError:
        lat = float(json.loads(event['body'])['lat'])
        lon = float(json.loads(event['body'])['lon'])
        event['out_dir'] = json.loads(event['body'])['out_dir']

    print event

    buffer_wgs84_geojson = geo_utils.pt_to_mill_buffer(lat, lon)

    # root dir for raster datasets
    data_root = 's3://palm-risk-poc/data'

    # our list of analyses we're interested in
    analysis_list = [('loss', 'wdpa'),
                     ('landcover', 'wdpa'),
                     ('primary', 'area')]

    event['geojson'] = buffer_wgs84_geojson

    for categorical_raster, area_raster in analysis_list:

        calc_name = '{}_{}'.format(categorical_raster, area_raster)

        if categorical_raster == 'loss':
            categorical_raster = r's3://gfw2-data/forest_change/hansen_2016_masked_30tcd/data.vrt'
        else:
            categorical_raster = '{}/{}/data.vrt'.format(data_root, categorical_raster)

        if area_raster == 'wdpa':
            area_raster = '{}/area_filtered_wdpa/data.vrt'.format(data_root)
        else:
            area_raster = 's3://gfw2-data/analyses/area_28m/data.vrt'

        # set proper layer_id and calc required
        func_config = event.copy()
        func_config['calc_name'] = calc_name
        func_config['categorical_raster'] = categorical_raster
        func_config['area_raster'] = area_raster

        print 'starting event for: '
        print func_config

        client.invoke(
            FunctionName='palm-risk-poc-dev-risk',
            InvocationType='Event',
            Payload=json.dumps(func_config)
            )


# analyze our buffered geom by a particular raster or two
def risk(event, context):

    # load geom from geojson string into shapely
    geom = shape(json.loads(event['geojson']))

    stats = geoprocessing.count_pairs(geom, [event['categorical_raster'], event['area_raster']])

    output_dict = {}

    # stats is a dict with keys of category_id::area_value and values of
    # pixel count. To unpack this:
    for key, pixel_count in stats.iteritems():
        category, area = key.split('::')

        category = int(float(category))
        area_ha = float(area) * float(pixel_count) / 10000.

        try:
            output_dict[category] += area_ha
        except KeyError:
            output_dict[category] = area_ha

    out_file = 'output/{}/{}.json'.format(event['out_dir'], event['calc_name'])
    s3.Bucket(bucket_uri).put_object(Key=out_file, Body=json.dumps(output_dict))


# to test locally
if __name__ == '__main__':
    d = {'queryStringParameters':
            {'lon': 112.8515625, 'lat': -2.19672724, 'out_dir': 'local-test'}
        }

    mill(d, None)
