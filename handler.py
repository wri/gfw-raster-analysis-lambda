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

    agg_values = params.get('aggregate_values', False)
    agg_by = params.get('aggregate_by')

    if agg_values in ['true', 'TRUE', 'True', True]:
        agg_values = True


    agg_list = ['week', 'month', 'year', 'quarter', 'all']

    if agg_by not in agg_list or agg_values != True:
        msg = 'For this batch service, aggregate_values must be True, and ' \
              'aggregate_by must be in {}'.format(', '.join(agg_list))
        return gfw_api.api_error(msg)

    analysis_raster = 's3://palm-risk-poc/data/glad/data.vrt'

    stats = geoprocessing.count(geom, analysis_raster)

    hist = util.unpack_glad_histogram(stats, agg_by)

    return gfw_api.serialize_glad(hist, area_ha, agg_by)


if __name__ == '__main__':
    aoi ={"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[117.164337416055,-0.146001213786356],[117.155703106841,-0.057749439573639],[117.130133243555,0.027110960960791],[117.088610223703,0.105318959360494],[117.032729390131,0.173868987816331],[116.9646379261,0.230126442445014],[116.886952514797,0.271929022470392],[116.802658856842,0.297669979066676],[116.714996872182,0.306360018259741],[116.62733601195,0.297665417784521],[116.543045531146,0.271920848997643],[116.465364792161,0.230116400643847],[116.397278664419,0.173859331577133],[116.341402857481,0.105312081565839],[116.299883592981,0.027108993740147],[116.274315418049,-0.057744991082702],[116.265680230058,-0.145989746845834],[116.274309760144,-0.234235059943345],[116.299872885489,-0.319090657019565],[116.341388234208,-0.397296240890491],[116.397261631207,-0.465846611406709],[116.465347025389,-0.522107112718315],[116.543028657773,-0.563914980303813],[116.627321403215,-0.589662660364852],[116.714985480799,-0.598359835271131],[116.80265112075,-0.589671696300947],[116.886948340271,-0.56393193102085],[116.964636750894,-0.522129896874692],[117.032730315149,-0.465872485244865],[117.088612191345,-0.397322188160817],[117.130135233987,-0.319113812726435],[117.155704320927,-0.234253104759971],[117.164337416055,-0.146001213786356]]]}}]}

    # why this crazy structure? Oh lambda . . . sometimes I wonder
    event = {
             'body': json.dumps({'geojson': aoi}),
             'queryStringParameters': {'aggregate_by':'year', 'aggregate_values': 'True', 'tile_id': '10N_00W'}
            }

    #glad_alerts(event, None)
    #analysis(event, None)
    #landcover(event, None)
    #loss_by_landcover(event, None)
    #umd_loss_gain(event, None)
    fire_analysis(event, None)
