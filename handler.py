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
    lat = float(event['queryStringParameters']['lat'])
    lon = float(event['queryStringParameters']['lon'])

    buffer_wgs84_geojson = geo_utils.pt_to_mill_buffer(lat, lon)

    # calculate approximate area per 0.00025 degree pixel
    # based on latitude of the mill center point
    pixel_area_m2 = geo_utils.lat_to_area_m2(lat)

    # root dir for raster datasets
    data_root = 's3://palm-risk-poc/data'

    layer_list = ['land_area', 'wdpa', 'peat', 'primary']

    event['pixel_area_m2'] = pixel_area_m2
    event['geojson'] = buffer_wgs84_geojson

    for layer in layer_list:
        for calc_type in ['loss', 'area']:
            raster_path = '{}/{}/data.vrt'.format(data_root, layer)

            # set proper layer_id and calc required
            func_config = event.copy()
            func_config['layer'] = layer
            func_config['raster_path'] = raster_path
            func_config['calc_type'] = calc_type

            client.invoke(
                FunctionName='palm-risk-poc-dev-risk',
                InvocationType='Event',
                Payload=json.dumps(func_config))


# analyze our buffered geom by a particular raster or two
def risk(event, context):

    # load geom from geojson string into shapely
    geom = shape(json.loads(event['geojson']))

    if event['calc_type'] == 'area':
        # returns a tuple of (total_pixels, zstats_dict)
        # return only the zstats dict to match the count_pairs output format
        stats = geoprocessing.count(geom, event['raster_path'])[1]

    else:
        loss_vrt = r's3://gfw2-data/forest_change/hansen_2016_masked_30tcd/data.vrt'
        stats = geoprocessing.count_pairs(geom, [loss_vrt, event['raster_path']])

    # convert pixel counts to area_ha
    area_stats = stats_to_area(stats, event['pixel_area_m2'])

    print area_stats

    out_file = 'output/{}/{}_{}.json'.format(event['out_dir'], event['layer'], event['calc_type'])
    s3.Bucket(bucket_uri).put_object(Key=out_file, Body=json.dumps(area_stats))


# multiply each count by an area coefficient based
# on the mill center point
def stats_to_area(input_dict, pixel_area_m2):
    area_dict = {}

    # update dictionary to point to area_ha, not pixel_count
    for raster_class, pixel_count in input_dict.iteritems():
        area_dict[raster_class] = pixel_count * pixel_area_m2 / 10000

    return area_dict


# to test locally
if __name__ == '__main__':
    d = {'queryStringParameters': {'lon': 112.8515625, 'lat': -2.19672724}}

    mill(d, None)
