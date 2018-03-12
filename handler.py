import json

import grequests
import sys

from geop import geo_utils, geoprocessing
from serializers import gfw_api
from utilities import util, lulc_util

glad_raster = 's3://palm-risk-poc/data/glad/data.vrt'


def umd_loss_gain(event, context):

    geom, area_ha = util.get_shapely_geom(event)
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
    #for analysis_type in ['loss', 'gain', 'extent']:
    for analysis_type in ['loss', 'extent', 'gain']:
        new_params = params.copy()
        new_params['analysis'] = analysis_type

        request_list.append(grequests.post(url, json=payload, params=new_params))

    # execute these requests in parallel
    response_list = grequests.map(request_list, size=3)

    return gfw_api.serialize_loss_gain(response_list, area_ha)


def analysis(event, context, analysis_raster=None, area_raster=None):

    geom, _ = util.get_shapely_geom(event)
    analysis_type = event['queryStringParameters']['analysis']
    thresh = event['queryStringParameters']['thresh']

    if not analysis_raster:
        ras_dict = {'loss': 's3://gfw2-data/forest_change/hansen_2016_masked_{}tcd/data.vrt'.format(thresh),
                    'extent': 's3://gfw2-data/forest_cover/2000_treecover/data.vrt',
                    'gain': 's3://gfw2-data/forest_change/tree_cover_gain/gaindata_2012/data.vrt'}

        area_raster = 's3://gfw2-data/analyses/area_28m/data.vrt'

        analysis_raster = ras_dict[analysis_type]

    stats = geoprocessing.count_pairs(geom, [analysis_raster, area_raster])

    # unpack the response from the gp function to standard {year: area} dict
    hist = util.unpack_count_histogram(analysis_type, stats)

    return gfw_api.serialize_analysis(hist, event)


def landcover(event, context):

    geom, area_ha = util.get_shapely_geom(event)

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

    geom, area_ha = util.get_shapely_geom(event)

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


def glad_alerts(event, context, ras=glad_raster):

    geom, area_ha = util.get_shapely_geom(event)
    payload = {'geojson': json.loads(event['body'])['geojson']}

    try:
        params = util.validate_glad_params(event)
    except ValueError, e:
        return gfw_api.api_error(str(e))

    stats = geoprocessing.count(geom, ras)

    hist = util.unpack_glad_histogram(stats, params)

    return gfw_api.serialize_glad(hist, area_ha, params['aggregate_by'], params['period'])


if __name__ == '__main__':

    aoi = {"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[100.05523681640625,62.710684552498556],[99.964599609375,62.63250704195784],[100.1678466796875,62.629981748883736],[100.05523681640625,62.710684552498556]]]}}]}
    #aoi = {"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[22.241821289062496,2.5260367521718403],[22.1319580078125,1.9771465537125772],[22.664794921874996,1.9716566508363325],[22.8570556640625,2.4711573377481715],[22.241821289062496,2.5260367521718403]]]}}]}
    aoi = {"type": "FeatureCollection", "features": [{"geometry": {"type": "Polygon", "coordinates": [[[-64.01513089930181, -10.250867673952202], [-64.01547724159283, -10.255188574409503], [-64.01627009532785, -10.259456511252596], [-64.01750186316461, -10.263630379052934], [-64.0191607173417, -10.267669976817634], [-64.02123071224656, -10.2715363953137], [-64.02369193682378, -10.2751923920038], [-64.0265207054007, -10.27860274997372], [-64.02968978512828, -10.281734617385585], [-64.03316865787271, -10.28455782417545], [-64.03692381404983, -10.2870451729338], [-64.04091907557586, -10.289172701154769], [-64.04511594481502, -10.290919912316015], [-64.04947397614296, -10.292269973551893], [-64.0539511665169, -10.293209878004708], [-64.05850436124986, -10.293730570282827], [-64.0630896710332, -10.293827033807522], [-64.06766289613611, -10.293498339204145], [-64.07217995363852, -10.29274765326688], [-64.07659730352252, -10.291582208411123], [-64.08087236945916, -10.290013232909372], [-64.08496395018159, -10.288055842586774], [-64.08883261743107, -10.28572889502616], [-64.092441096599, -10.283054807694453], [-64.09575462636374, -10.280059341754091], [-64.09874129383302, -10.276771353650766], [-64.10137234195015, -10.273222516883683], [-64.10362244620131, -10.269447016649996], [-64.10546995796813, -10.265481220316367], [-64.10689711220245, -10.261363326903165], [-64.10789019745539, -10.2571329989683], [-64.1084396866631, -10.252830980443619], [-64.10854032747912, -10.24849870411266], [-64.1081911913373, -10.244177892515113], [-64.10739568083196, -10.239910156123585], [-64.1061614954035, -10.235736592663397], [-64.10450055572001, -10.23169739143185], [-64.1024288875389, -10.2278314464234], [-64.09996646621865, -10.224175981979519], [-64.0971370234221, -10.22076619456048], [-64.09396781790822, -10.217634914078262], [-64.09048937264474, -10.214812288041438], [-64.08673518078797, -10.212325491541488], [-64.08274138336343, -10.210198465861323], [-64.07854642174233, -10.208451688210761], [-64.07419066824033, -10.207101974796126], [-64.06971603836466, -10.206162319108248], [-64.0651655884037, -10.20564176697906], [-64.06058310218604, -10.205545329601257], [-64.05601267093442, -10.20587393534477], [-64.05149827020276, -10.206624420830288], [-64.0470833379107, -10.20778956134566], [-64.0428103574805, -10.209358140314016], [-64.03872045003418, -10.211315057148683], [-64.0348529795274, -10.213641472462912], [-64.0312451745786, -10.216314989243612], [-64.02793177059962, -10.219309868254328], [-64.02494467564891, -10.222597275604668], [-64.02231266321103, -10.226145560113492], [-64.02006109485808, -10.229920557806684], [-64.01821167547318, -10.233885920629877], [-64.01678224341305, -10.23800346622217], [-64.01578659766221, -10.242233545392244], [-64.01523436368298, -10.24653542376819], [-64.01513089930181, -10.250867673952202]]]}, "type": "Feature", "properties": {}}]}

    # why this crazy structure? Oh lambda . . . sometimes I wonder
    event = {
             'body': json.dumps({'geojson': aoi}),
             'queryStringParameters': {'gladConfirmOnly': True, 'thresh': '30', 'period':'2016-01-01,2017-01-01'}
            }


    glad_alerts(event, None)
    #analysis(event, None, loss_raster, area_raster)
    #landcover(event, None)
    #loss_by_landcover(event, None)
    #umd_loss_gain(event, None)
