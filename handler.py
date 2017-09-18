import json
import requests
import pyproj
import subprocess
import boto3
import uuid

from shapely.geometry import mapping, Point, shape
from shapely.ops import cascaded_union
from geop import geo_utils, geoprocessing

s3 = boto3.resource('s3')


def mill(event, context):
    lat = event['queryStringParameters']['lat']
    lon = event['queryStringParameters']['lon']

    wgs84_str = 'EPSG:4326'
    eckVI_str = "esri:54010"

    eckVI_proj = pyproj.Proj(init="{}".format(eckVI_str))
    eck_point = Point(eckVI_proj(lon, lat))

    buffer_eck = eck_point.buffer(50000)
    buffer_wgs84 = geo_utils.reproject(buffer_eck, eckVI_str, wgs84_str)

    total_area_ha = buffer_eck.area / 10000

    # layer_dict = {'gadm28': 'gadm28_adm0.shp',
    #               'wdpa': 'wdpa_protected_areas.shp',
    #               'peat': 'idn_peat_lands.shp',
    #               'primary_forest': 'idn_primary_forest_shp.shp'}

    layer_dict = {'tree-cover-dummy-layer':
                  's3://gfw2-data/forest_cover/2000_treecover/data.vrt'}

    mill_dict = {}

    for layer_id, layer_path in layer_dict.iteritems():

        mill_dict[layer_id] = {}

        # simulate calculating 5 layers in total
        for i in range(0, 5):
            print 'calculating stats against 2000 - 2016 loss'
            loss_dict = calc_loss(buffer_wgs84, layer_path)

            mill_dict[layer_id]['loss'] = loss_dict

    out_file = str(uuid.uuid4()) + '.json'
    s3.Bucket('palm-risk-poc').put_object(Key=out_file, Body=json.dumps(mill_dict))

    return {
    'statusCode': 200,
    'headers': {
        'Access-Control-Allow-Origin': '*'
    },
    'body': json.dumps({
        'output_file': r's3://palm-risk-poc/{}'.format(out_file),
        'buffer_area_ha': total_area_ha
    })
    }


def calc_loss(aoi_intersect, input_ras):

    loss_vrt = r's3://gfw2-data/forest_change/hansen_2016_masked_30tcd/data.vrt'

    raster_dict = geoprocessing.count_pairs(aoi_intersect, [loss_vrt, input_ras])

    print raster_dict

    return raster_dict


if __name__ == '__main__':
    d = {'queryStringParameters': {'lon': 112.8515625, 'lat': -2.19672724}}

    mill(d, None)
