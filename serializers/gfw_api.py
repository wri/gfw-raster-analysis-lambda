import json

from shapely.geometry import shape


def serialize_loss_gain(response_list, aoi_polygon_area):

    # response list is three dictionaries
    # that look like {'extent': 23421}
    # need to combine them so we can use the keys to extract proper values
    response_dict = dict([d.json().items()[0] for d in response_list])

    print response_dict

    serialized = {'data': {
                    'attributes': {
                        'areaHa': aoi_polygon_area,
                        'gain': response_dict['gain'],
                        'loss': response_dict['loss'],
                        'treeExtent': response_dict['extent']
                        },
                    'id': None,
                    'type': 'umd'
                    }
                }

    return http_response(serialized)


def serialize_analysis(analysis_type, hist, event):

    if analysis_type == 'loss':
        return serialize_loss(hist, event)

    else:
        return serialize_extent_or_gain(analysis_type, hist)

def serialize_extent_or_gain(analysis_type, hist):

    print analysis_type
    print hist

    area_total = 0

    for pixel_value, area_ha in hist.iteritems():

        if analysis_type == 'extent' and int(pixel_value) >= 30:
            area_total += area_ha
        elif analysis_type == 'gain' and int(pixel_value) == 1:
            area_total += area_ha

    return http_response({analysis_type: area_total})


def serialize_loss(loss_area_dict, event):

    params = event['queryStringParameters']

    if params:
        period = params.get('period', None)
        aggregate_values = params.get('aggregate_values', None)
    else:
        period = None
        aggregate_values = None

    # filter by period if given
    if period:
        date_min, date_max = period.split(',')
        year_min = int(date_min.split('-')[0])
        year_max = int(date_max.split('-')[0])

        loss_area_dict = dict((k, v) for k, v in loss_area_dict.iteritems() if k >= year_min and k <= year_max)

    # if we want year-by-year results . . .
    if (aggregate_values and aggregate_values.lower() == 'false') or (aggregate_values == False):
        loss = loss_area_dict

    else:
        loss = sum(loss_area_dict.values())

    return http_response({'loss': loss})


def serialize_landcover(landcover_dict, event):

    landcover_list = []
    lkp = build_globcover_lookup()

    for lulc_val, area_ha in landcover_dict.iteritems():

        landcover_list.append({
        'className': lkp[lulc_val],
        'classVal': lulc_val,
        'result': area_ha
        })

    serialized = {'data': {
                    'attributes': {
                        'areaHa': -9999,
                        'landcover': landcover_list
                    }
    }}

    return http_response(serialized)


def http_response(response):

    print response

    return {
        'statusCode': 200,
        'headers': {'Access-Control-Allow-Origin': '*'},
        'body': json.dumps(response)
            }


def build_globcover_lookup():

    return {
        1: 'Agriculture',
        2: 'Forest',
        3: 'Grassland',
        4: 'Wetland',
        5: 'Settlement',
        6: 'Shrubland',
        7: 'Sparse vegetation',
        8: 'Bare',
        9: 'Water',
        10: 'Permanent snow and ice'
        }
