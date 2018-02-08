import json

import grequests

from geop import geo_utils, geoprocessing
from serializers import gfw_api
from utilities import util, lulc_util


def umd_loss_gain(event, context):

    geom = util.get_shapely_geom(event)
    area_ha = geo_utils.get_polygon_area(geom)

    payload = {'geojson': json.loads(event['body'])['geojson']}

    params = event.get('queryStringParameters')
    if not params:
        params = {}

    thresh = int(params.get('thresh', 30))
    params['thresh'] = thresh

    valid_thresh = [10, 30, 90]

    if thresh not in valid_thresh:
        thresh_str = ', '.join([str(x) for x in valid_thresh])
        msg = 'thresh {} supplied, for this S3 endpoint must be one of {}'.format(thresh, thresh_str)
        return gfw_api.api_error(msg)

    url = 'https://0yvx7602sb.execute-api.us-east-1.amazonaws.com/dev/analysis'
    request_list = []

    # add specific analysis type for each request
    for analysis_type in ['loss', 'gain', 'extent']:

        new_params = params.copy()
        new_params['analysis'] = analysis_type
        print new_params
        request_list.append(grequests.post(url, json=payload, params=new_params))

    # execute these requests in parallel
    response_list = grequests.map(request_list, size=3)

    return gfw_api.serialize_loss_gain(response_list, area_ha)


def analysis(event, context):

    geom = util.get_shapely_geom(event)
    analysis_type = event['queryStringParameters']['analysis']
    thresh = event['queryStringParameters']['thresh']

    ras_dict = {'loss': 's3://gfw2-data/forest_change/hansen_2016_masked_{}tcd/data.vrt'.format(thresh),
                'extent': 's3://gfw2-data/forest_cover/2000_treecover/data.vrt',
                'gain': 's3://gfw2-data/forest_change/tree_cover_gain/gaindata_2012/data.vrt'}

    analysis_raster = ras_dict[analysis_type]
    area_raster = 's3://gfw2-data/analyses/area_28m/data.vrt'

    stats = geoprocessing.count_pairs(geom, [analysis_raster, area_raster])

    # unpack the response from the gp function to standard {year: area} dict
    hist = util.unpack_count_histogram(analysis_type, stats)

    return gfw_api.serialize_analysis(hist, event)


def fire_analysis(event, context):

    geom = util.get_shapely_geom(event)
    tile_id = event['queryStringParameters']['tile_id']

    date_list = geoprocessing.point_stats(geom, tile_id) # looks like [u'2016-05-09', u'2016-05-13', u'2016-06-03', u'2016-05-07', u'2016-05-07']

    # makes json formatted info of tile_id: date list
    return gfw_api.serialize_fire_analysis(date_list, tile_id)


def landcover(event, context):

    geom = util.get_shapely_geom(event)
    area_ha = geo_utils.get_polygon_area(geom)

    params = event['queryStringParameters']
    if not params:
        params = {}

    layer_name = params.get('layer')

    valid_layers = lulc_util.get_valid_layers()

    if layer_name not in valid_layers:
        msg = 'Layer query param must be one of: {}'.format(', '.join(valid_layers))
        return gfw_api.api_error(msg)

    lulc_raster = lulc_util.ras_lkp(layer_name)
    area_raster = 's3://gfw2-data/analyses/area_28m/data.vrt'
    stats = geoprocessing.count_pairs(geom, [lulc_raster, area_raster])

    hist = util.unpack_count_histogram('landcover', stats)

    return gfw_api.serialize_landcover(hist, layer_name, area_ha)


def loss_by_landcover(event, context):

    geom = util.get_shapely_geom(event)
    area_ha = geo_utils.get_polygon_area(geom)

    params = event['queryStringParameters']
    if not params:
        params = {}

    layer_name = params.get('layer')

    valid_layers = lulc_util.get_valid_layers()

    if layer_name not in valid_layers:
        msg = 'Layer query param must be one of: {}'.format(', '.join(valid_layers))
        return gfw_api.api_error(msg)

    lulc_raster = lulc_util.ras_lkp(layer_name)
    loss_raster = 's3://gfw2-data/forest_change/hansen_2016_masked_30tcd/data.vrt'
    area_raster = 's3://gfw2-data/analyses/area_28m/data.vrt'

    raster_list = [lulc_raster, loss_raster, area_raster]
    stats = geoprocessing.count_pairs(geom, raster_list)

    hist = util.unpack_count_histogram('loss-by-landcover', stats)

    return gfw_api.serialize_loss_by_landcover(hist, area_ha, event)


def fire_alerts(event, context):

    geom = util.get_shapely_geom(event)
    area_ha = geo_utils.get_polygon_area(geom)

    payload = {'geojson': json.loads(event['body'])['geojson']}

    params = event.get('queryStringParameters')
    if not params:
        params = {}

    # send list of tiles to another enpoint called fire_analysis(geom, tile)
    url = 'https://0yvx7602sb.execute-api.us-east-1.amazonaws.com/dev/fire_analysis'
    request_list = []

    # get list of tiles that intersect the aoi
    tiles = geoprocessing.find_tiles(geom)

    # add specific analysis type for each request
    for tile in tiles:

        new_params = params.copy()
        new_params['tile_id'] = tile

        request_list.append(grequests.post(url, json=payload, params=new_params))

    # execute these requests in parallel
    response_list = grequests.map(request_list, size=len(tiles))

    # what should i return here? maybe call serialize_glad -> serialize glad_or_fires?
    return gfw_api.serialize_glad(response_list, area_ha)


def glad_alerts(event, context):

    geom = util.get_shapely_geom(event)
    area_ha = geo_utils.get_polygon_area(geom)

    payload = {'geojson': json.loads(event['body'])['geojson']}
    params = event.get('queryStringParameters')

    if not params:
        params = {}

    # do not allow user to query this because api doens't recognize it
    if params.get('period'):
        msg = "This api does not filter GLAD by date. Please remove the 'period' parameter"
        return gfw_api.api_error(msg)


    agg_values = params.get('aggregate_values', False)
    agg_by = params.get('aggregate_by')

    if agg_values in ['true', 'TRUE', 'True', True]:
        agg_values = True


    agg_list = ['day', 'week', 'month', 'year', 'quarter', 'all']

    if agg_by not in agg_list or agg_values != True:
        msg = 'For this batch service, aggregate_values must be True, and ' \
              'aggregate_by must be in {}'.format(', '.join(agg_list))
        return gfw_api.api_error(msg)

    analysis_raster = 's3://palm-risk-poc/data/glad/data.vrt'

    stats = geoprocessing.count(geom, analysis_raster)

    hist = util.unpack_glad_histogram(stats, agg_by)

    return gfw_api.serialize_glad(hist, area_ha, agg_by)


if __name__ == '__main__':
    aoi ={"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"MultiPolygon","coordinates":[[[[21.0723461722907,-3.17077555929246],[21.0710301780474,-3.09152122482142],[21.1729995906205,-3.09152122482142],[21.1743219533783,-3.17077555929246],[21.0723461722907,-3.17077555929246]]]]}}]}

    # why this crazy structure? Oh lambda . . . sometimes I wonder
    event = {
             'body': json.dumps({'geojson': aoi}),
             'queryStringParameters': {'aggregate_by':'all', 'aggregate_values': 'True', 'tile_id': '10N_00W'}
            }

    glad_alerts(event, None)
    #analysis(event, None)
    #landcover(event, None)
    #loss_by_landcover(event, None)
    #umd_loss_gain(event, None)
    #fire_analysis(event, None)
