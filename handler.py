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

    thresh = int(params.get('thresh', 30))
    valid_thresh = [10, 30, 90]

    if thresh not in valid_thresh:
        thresh_str = ', '.join([str(x) for x in valid_thresh])
        msg = 'thresh {} supplied, for this S3 endpoint must be one of {}'.format(thresh, thresh_str)
        return gfw_api.api_error(msg) 

    if not params:
        params = {'thresh': thresh}

    url = 'https://0yvx7602sb.execute-api.us-east-1.amazonaws.com/dev/analysis'
    request_list = []

    # add specific analysis type for each request
    for analysis_type in ['loss', 'gain', 'extent']:

        new_params = params.copy()
        new_params['analysis'] = analysis_type

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


def landcover(event, context):

    geom = util.get_shapely_geom(event)
    area_ha = geo_utils.get_polygon_area(geom)

    layer_name = event['queryStringParameters'].get('layer')

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

    layer_name = event['queryStringParameters'].get('layer')
    
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


def glad(event, context):

    geom = util.get_shapely_geom(event)
    area_ha = geo_utils.get_polygon_area(geom)

    payload = {'geojson': json.loads(event['body'])['geojson']}
    params = event.get('queryStringParameters')

    if not params:
        params = {}

    agg_values = params.get('aggregate_values', 'false').lower()
    agg_by = params.get('aggregate_by')

    agg_list = ['week', 'month', 'year', 'quarter', 'all']

    if agg_by not in agg_list or agg_values != 'true':
        msg = 'For this batch service, aggregate_values must be True, and ' \
              'agg_by must be in {}'.format(', '.join(agg_list)) 
        return gfw_api.api_error(msg)

    analysis_raster = 's3://palm-risk-poc/data/glad/data.vrt'

    count, stats = geoprocessing.count(geom, analysis_raster)

    hist = util.unpack_glad_histogram(stats, agg_by)

    return gfw_api.serialize_glad(hist, area_ha, agg_by)


if __name__ == '__main__':
    aoi = {"type": "FeatureCollection","features": [{"type": "Feature","properties": {},"geometry": {"type": "Polygon", "coordinates": [[[112.52347809425214, -2.4629252286881287], [112.52375417593427, -2.5017628816946242], [112.51916147519748, -2.5402267045362503], [112.50974204699374, -2.577946263970524], [112.49558330886562, -2.614558288658496], [112.47681732740618, -2.6497101679925885], [112.45361971343202, -2.6830633484981927], [112.42620812990329, -2.7142965950846145], [112.39484041908888, -2.7431090857103273], [112.35981235819803, -2.7692233096203624], [112.3214550556943, -2.792387741195524], [112.28013200376171, -2.8123792636063976], [112.23623580588652, -2.8290053188698887], [112.19018460220374, -2.842105763539243], [112.1424182190733, -2.851554412095351], [112.09339407322184, -2.857260253120008], [112.04358286462103, -2.8591683264916665], [111.9934640959716, -2.857260253120008], [111.94352146012484, -2.851554412095351], [111.894238139886, -2.8421057635392435], [111.84609206731211, -2.829005318869889], [111.79955119173738, -2.8123792636063984], [111.75506880725054, -2.792387741195524], [111.71307899113648, -2.769223309620364], [111.6739922048242, -2.7431090857103277], [111.6381911081168, -2.714296595084616], [111.60602663590022, -2.6830633484981936], [111.57781438413839, -2.6497101679925903], [111.55383134878588, -2.614558288658497], [111.53431305732953, -2.5779462639705244], [111.51945112806735, -2.5402267045362517], [111.50939128702453, -2.5017628816946242], [111.50423186668532, -2.462925228688129], [111.50402280458742, -2.4240877731073645], [111.50876515339806, -2.3856245349570426], [111.5184111074848, -2.3479059250133516], [111.53286454433317, -2.3112951781314286], [111.55198207256961, -2.276144855818887], [111.57557457194386, -2.2427934517202726], [111.60340920452126, -2.211562132664538], [111.63521187064549, -2.182751646622346], [111.6706700780462, -2.15663942731472], [111.70943618787126, -2.1334769233240363], [111.75113099748947, -2.113487177400656], [111.7953476166845, -2.096862679253675], [111.8416555913836, -2.0837635124853295], [111.88960522734892, -2.0743158135003075], [111.9387320653001, -2.0686105572207465], [111.98856145871547, -2.066702681293707], [112.03861320603454, -2.068610557220746], [112.08840619009754, -2.074315813500307], [112.13746297934445, -2.083763512485329], [112.185314347468, -2.096862679253675], [112.23150367078676, -2.1134871774006547], [112.27559116547985, -2.133476923324035], [112.3171579299057, -2.1566394273147185], [112.35580976042468, -2.1827516466223447], [112.39118071236838, -2.2115621326645356], [112.42293638097355, -2.2427934517202703], [112.45077688015866, -2.2761448558188837], [112.47443949992143, -2.311295178131426], [112.49370102584041, -2.3479059250133494], [112.50837970666139, -2.3856245349570395], [112.51833685824398, -2.4240877731073605], [112.52347809425214, -2.4629252286881256], [112.52347809425214, -2.4629252286881287]]]}}]}

    # why this crazy structure? Oh lambda . . . sometimes I wonder
    event = {
             'body': json.dumps({'geojson': aoi}),
             'queryStringParameters': {'aggregate_values': 'true',
                                       'layer': 'primary-forest',
                                       'aggregate_by': 'week'}
            }

    # glad(event, None)
    # analysis(event, None)
    landcover(event, None)
    # loss_by_landcover(event, None)
