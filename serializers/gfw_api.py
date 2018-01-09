import json


def serialize_loss_gain(response_list, aoi_polygon_area):

    # response list is three dictionaries
    # that look like {'extent': 23421}
    # need to combine them so we can use the keys to extract proper values
    response_dict = dict([d.json().items()[0] for d in response_list])

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

        # for extent, our the 30% threshold is actually >30
        if analysis_type == 'extent' and int(pixel_value) > 30:
            area_total += area_ha
        elif analysis_type == 'gain' and int(pixel_value) == 1:
            area_total += area_ha

    return http_response({analysis_type: area_total})


def serialize_loss(loss_area_dict, event):

    params = event['queryStringParameters']
    period = params.get('period', None)
    aggregate_values = params.get('aggregate_values', True)

    if isinstance(aggregate_values, (unicode, str)) and aggregate_values.lower() == 'false':
        aggregate_values = False

    # filter by period if given
    if period:
        date_min, date_max = period.split(',')
        year_min = int(date_min.split('-')[0])
        year_max = int(date_max.split('-')[0])

    else:
        year_min = 2001
        year_max = 2016

    requested_years = range(year_min, year_max + 1)
    empty_year_dict = {year: 0 for year in requested_years}

    loss_area_dict = dict((k, v) for k, v in loss_area_dict.iteritems() if k >= year_min and k <= year_max)

    # if we want year-by-year results . . .
    if aggregate_values:
        loss = sum(loss_area_dict.values())

    # populate data where we have it, leaving the other values empty
    else:
        for year, area_ha in loss_area_dict.iteritems():
            empty_year_dict[year] = area_ha

        loss = empty_year_dict

    return http_response({'loss': loss})


def serialize_landcover(landcover_dict, input_poly_area):

    landcover_list = []
    lkp = build_globcover_lookup()

    for lulc_val, area_ha in landcover_dict.iteritems():

        landcover_list.append({
        'className': lkp[lulc_val],
        'classVal': str(lulc_val),
        'result': area_ha,
        'resultType': 'areaHectares'
        })

    serialized = {'data': {
                    'attributes': {
                        'areaHa': input_poly_area,
                        'landcover': landcover_list
                        },
                    'type': 'gfw-landcover-2015',
                    'id': None
    }}

    return http_response(serialized)


def serialize_loss_by_landcover(hist, input_poly_area, event):

    period = event['queryStringParameters'].get('period', None)

    # filter by period if given
    if period:
        date_min, date_max = period.split(',')
        year_min = int(date_min.split('-')[0])
        year_max = int(date_max.split('-')[0])
    else:
        year_min = 2001
        year_max = 2016

    requested_years = range(year_min, year_max + 1)

    lulc_vals = [0, 1]
    empty_year_dict = {year: 0 for year in requested_years}
    final_dict = {str(val): empty_year_dict.copy() for val in lulc_vals}
    print final_dict

    for combine_value, area_ha in hist.iteritems():

        lulc = str(int(combine_value) / 500)
        year = 2000 + int(combine_value) % 500

        if year in requested_years:
            final_dict[lulc][year] = area_ha

    histogram = lookup(final_dict, False)

    serialized = {'data': {
                    'attributes': {
                        'areaHa': input_poly_area,
                        'histogram': histogram
                        },
                    'type': 'primary-forest',
                    'id': None
                    }
                }

    return http_response(serialized)


def http_response(response):

    print json.dumps(response, indent=4, sort_keys=True)

    return {
        'statusCode': 200,
        'headers': {'Access-Control-Allow-Origin': '*'},
        'body': json.dumps(response)
            }


def api_error(msg):
    print msg

    return {
        'statusCode': 400,
        'headers': {'Access-Control-Allow-Origin': '*'},
        'body': json.dumps({'error': msg})
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


def lookup(result_dict, count_pixels):

    lkp = {'0': 'Not Primary Forest', '1': 'Primary Forest'}

    if count_pixels:
        area_type = 'pixelCount'
    else:
        area_type = 'areaHectares'

    output_list = []

    for key, val in result_dict.items():

        landcover_name = lkp[key]

        result_dict = {'resultType': area_type,
                    'className': landcover_name,
                    'classVal': key}

        # unpack year dict into array of dicts
        if isinstance(val, dict):
            val = [{'year': year, 'result': result} for year, result in val.items()]

        result_dict['result'] = val

        output_list.append(result_dict)

    return output_list


def serialize_glad(hist, area_ha, agg_by):

    serialized = {
    "data": {
        "aggregate_by": agg_by, 
        "aggregate_values": True,
        "attributes": {
            "downloadUrls": None,
            "value": hist},
        "period": None,
        "type": "glad-alerts"}
    }

    return http_response(serialized)


 
