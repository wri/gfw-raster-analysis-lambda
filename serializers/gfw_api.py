import json


def serialize_loss_gain(loss_area_dict, event):

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

    serialized = {'data': {
                    'attributes': {
                        'areaHa': -9999,
                        'gain': -9999,
                        'loss': loss,
                        'treeExtent': -9999
                        },
                    'id': None,
                    'type': 'umd'
                    }
                }

    return http_response(serialized)


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
