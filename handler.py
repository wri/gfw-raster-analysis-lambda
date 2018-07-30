import json
import os
import sys


# add path to included packages
# these are all stored in the root of the zipped deployment package
root_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, root_dir)

from geop import geo_utils, geoprocessing
from serializers import gfw_api
from utilities import util, lulc_util

data_dir = os.path.join(root_dir, 'data')
glad_raster = os.path.join(data_dir, 'glad.vrt')

# temporary - to log s3 latency
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def analysis(event, context, analysis_raster=None, area_raster=None):

    geom, _ = util.get_shapely_geom(event)
    analysis_type = event['queryStringParameters']['analysis']
    thresh = event['queryStringParameters']['thresh']

    util_dir = os.path.join(root_dir, 'utilities')

    if not analysis_raster:
        ras_dict = {'loss': os.path.join(data_dir, 'loss{}.vrt'.format(thresh)),
                    'extent': os.path.join(data_dir, 'extent.vrt'),
                    'gain': os.path.join(data_dir, 'gain.vrt')}

        area_raster = os.path.join(data_dir, 'area.vrt')

        analysis_raster = ras_dict[analysis_type]

    stats = geoprocessing.count_pairs(geom, [analysis_raster, area_raster])

    # unpack the response from the gp function to standard {year: area} dict
    hist = util.unpack_count_histogram(analysis_type, stats)

    return gfw_api.serialize_analysis(hist, event)


def landcover(event, context):

    geom, area_ha = util.get_shapely_geom(event)
    params = event['queryStringParameters'] if event['queryStringParameters'] else {}

    layer_name = params.get('layer')

    valid_layers = lulc_util.get_valid_layers()

    if layer_name not in valid_layers:
        msg = 'Layer query param must be one of: {}'.format(', '.join(valid_layers))
        return gfw_api.api_error(msg)

    lulc_raster = lulc_util.ras_lkp(layer_name)
    area_raster = os.path.join(data_dir, 'area.vrt')
    stats = geoprocessing.count_pairs(geom, [lulc_raster, area_raster])

    hist = util.unpack_count_histogram('landcover', stats)

    return gfw_api.serialize_landcover(hist, layer_name, area_ha)


def extent_by_landcover(event, context, lulc_raster=None, extent_raster=None, area_raster=None):

    geom, area_ha = util.get_shapely_geom(event)

    params = event['queryStringParameters'] if event['queryStringParameters'] else {}

    layer_name = params.get('layer')

    valid_layers = lulc_util.get_valid_layers()

    if layer_name not in valid_layers:
        msg = 'Layer query param must be one of: {}'.format(', '.join(valid_layers))
        return gfw_api.api_error(msg)

    if not lulc_raster:
        lulc_raster = lulc_util.ras_lkp(layer_name)
        extent_raster = os.path.join(data_dir, 'extent.vrt')
        area_raster = os.path.join(data_dir, 'area.vrt')

    raster_list = [lulc_raster, extent_raster, area_raster]
    stats = geoprocessing.count_pairs(geom, raster_list)

    hist = util.unpack_count_histogram('extent-by-landcover', stats)

    return gfw_api.serialize_extent_by_landcover(hist, layer_name, area_ha, event)


def loss_by_landcover(event, context):

    geom, area_ha = util.get_shapely_geom(event)
    params = event['queryStringParameters'] if event['queryStringParameters'] else {}

    layer_name = params.get('layer')

    valid_layers = lulc_util.get_valid_layers()

    if layer_name not in valid_layers:
        msg = 'Layer query param must be one of: {}'.format(', '.join(valid_layers))
        return gfw_api.api_error(msg)

    lulc_raster = lulc_util.ras_lkp(layer_name)
    loss_raster = os.path.join(data_dir, 'loss30.vrt')
    area_raster = os.path.join(data_dir, 'area.vrt')

    raster_list = [lulc_raster, loss_raster, area_raster]
    stats = geoprocessing.count_pairs(geom, raster_list)

    hist = util.unpack_count_histogram('loss-by-landcover', stats)

    return gfw_api.serialize_loss_by_landcover(hist, area_ha, event)


def glad_alerts(event, context, ras=glad_raster):

    geom, area_ha = util.get_shapely_geom(event)

    try:
        params = util.validate_glad_params(event)
    except ValueError, e:
        return gfw_api.api_error(str(e))

    stats = geoprocessing.count(geom, ras)

    hist = util.unpack_glad_histogram(stats, params)

    return gfw_api.serialize_glad(hist, area_ha, params['aggregate_by'], params['period'])


if __name__ == '__main__':

    aoi = {"features":[{"properties":{},"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[140.2137,-6.3999],[140.3078,-6.3685],[140.3249,-6.3924],[140.3249,-6.4606],[140.3064,-6.49],[140.2274,-6.4838],[140.2068,-6.434],[140.2137,-6.3999]]]}}],"crs":{},"type":"FeatureCollection"}

    # why this crazy structure? Oh lambda . . . sometimes I wonder
    event = {
        'body': json.dumps({'geojson': aoi}),
        'queryStringParameters': {'thresh': 30, 'aggregate_values': False, 'analysis': 'extent'}
    }


    print analysis(event, None)

