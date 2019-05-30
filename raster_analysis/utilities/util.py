import json
import datetime
from collections import defaultdict
import sys
from functools import partial

import pyproj
from shapely.geometry import shape
from shapely.ops import transform
from flask import request

from errors import Error


def get_shapely_geom():

    geojson = request.get_json().get('geojson', None) if request.get_json() else None

    if len(geojson['features']) > 1:
        raise Error('Currently accepting only 1 feature at a time')

    # grab the actual geometry-- that's the level on which shapely operates
    geom = shape(geojson['features'][0]['geometry'])
    area_ha = get_polygon_area(geom)

    if area_ha > 10000000:
        raise Error('Geometry is too large for custom stats/download. ' \
                    'Please try again with a smaller area of interest')

    return geom, area_ha


def create_resp_dict(date_dict):
    k = date_dict.keys() # alert date = datetime.datetime(2015, 6, 4, 0, 0)
    v = date_dict.values() # count

    resp_dict = {
                 'year': grouped_and_to_rows([x.year for x in k], v, 'year'),
                 # month --> quarter calc: https://stackoverflow.com/questions/1406131
                 'quarter': grouped_and_to_rows([(x.year, (x.month-1)//3 + 1) for x in k], v, 'quarter'),
                 'month':  grouped_and_to_rows([(x.year, x.month) for x in k], v, 'month'),
                 'week': grouped_and_to_rows([(x.year, x.isocalendar()[1]) for x in k], v, 'week'),
                 'day': grouped_and_to_rows([(x.year, x.strftime('%Y-%m-%d')) for x in k], v, 'day'),
                 'total': sum(v)
                }

    return resp_dict


def unpack_glad_histogram(stats, params):

    # if there are no glad alerts, give it something to work with so response is empty list
    if stats == {}:
        stats = {'0000':0}

    hist_type = params['aggregate_by']
    period = params['period']

    date_dict = {}

    for conf_days, count in stats.iteritems():

        alert_date, conf = glad_val_to_date_conf(conf_days)

        valid_pixel = check_conf(conf, params)

        if valid_pixel:
            try:
                date_dict[alert_date] += count # if date exists, add count to id
            except KeyError:
                date_dict[alert_date] = count # if date doesn't exist, create it, set equal to count

    # filter dates by period
    start_date, end_date = period_to_dates(period)
    filtered_by_period = {alert_date : count for alert_date, count in date_dict.iteritems() if start_date <= alert_date <= end_date}

    resp_dict = create_resp_dict(filtered_by_period)

    return resp_dict.get(hist_type, {'all': resp_dict})


def check_conf(conf_val, params):

    valid_pixel = False

    # if we're filtering by confidence, only select conf values of 3
    if params['gladConfirmOnly']:
        if int(conf_val) == 3:
            valid_pixel = True
    else:
        valid_pixel = True

    return valid_pixel	

    
def glad_val_to_date_conf(glad_val):

    glad_str = str(glad_val)

    total_days = int(glad_str[1:])
    year = total_days / 365 + 2015
    julian_day = total_days % 365

    conf = int(glad_str[0]) 

    # https://stackoverflow.com/questions/17216169/
    alert_date = datetime.datetime(year, 1, 1) + datetime.timedelta(julian_day - 1)

    return alert_date, conf


def grouped_and_to_rows(keys, vals, agg_type):

    # source: https://jakevdp.github.io/blog/2017/03/22/group-by-from-scratch/
    count = defaultdict(int)
    for key, val in zip(keys, vals):
        count[key] += val
    grouped = dict(count)

    final_list = []

    for key, val in grouped.iteritems():

        if agg_type == 'year':
	    row = {agg_type: key}
        else:
            row = {'year': key[0], agg_type: key[1]}
    
        # compatibility with old API
        if agg_type == 'day':
            row['alert_date'] = row['day']

        row['count'] = val
        final_list.append(row)

    return final_list

def set_default_period():
    today = datetime.datetime.now().date()
    today = today.strftime('%Y-%m-%d')

    return "2015-01-01,{}".format(today)

def validate_glad_params():

    # create empty params dict of final params
    params = {}

    # if period isn't supplied, set default
    period = request.args.get('period', set_default_period())

    params['period'] = period

    # make sure the supplied period is in the right format, dates make sense
    check_dates(period)

    agg_values = check_param_true(request.args.get('aggregate_values', False))
    glad_confirm_only = check_param_true(request.args.get('gladConfirmOnly', False))
    agg_by = request.args.get('aggregate_by', 'total')
    download_format = parse_download_format(request.args.get('format', 'csv'))

    agg_list = ['day', 'week', 'month', 'year', 'quarter', 'all']

    if agg_values and agg_by not in agg_list:
        msg = 'If aggregate_values is True,  ' \
              'aggregate_by must be one of {}'.format(', '.join(agg_list))
        raise Error(msg)

    params['aggregate_values'] = agg_values
    params['aggregate_by'] = agg_by
    params['gladConfirmOnly'] = glad_confirm_only
    params['format'] = download_format

    return params


def check_param_true(param):
    return param in ['true', 'TRUE', 'True', True]


def parse_download_format(fmt):
    if fmt not in ['csv', 'json']:
        raise Error("format must be one of ['csv', 'json']")

    return fmt


def check_dates(period):

    try:
        start_date, end_date = period_to_dates(period)
    except ValueError:
        raise Error('period must be formatted as YYYY-mm-dd,YYYY-mm-dd')

    if start_date > end_date:
        raise Error('Start date must be <= end date')

    earliest_date = datetime.datetime.strptime('2015-01-01', '%Y-%m-%d')
    if start_date < earliest_date:
        raise Error('Start date must be later than Jan 1, 2015')


def period_to_dates(period):

    start_date, end_date = period.split(',')
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    return start_date, end_date


def get_polygon_area(geom):

    # source: https://gis.stackexchange.com/a/166421/30899
    geom_area = transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat1=geom.bounds[1],
                lat2=geom.bounds[3])),
        geom)

    # return area in ha
    return geom_area.area / 10000.


def filter_rows(input_tuple, params):

    x, y, z = input_tuple

    period = params['period']
    start_date, end_date = period_to_dates(period)

    alert_date, conf = glad_val_to_date_conf(z)

    valid_conf = check_conf(conf, params)

    if start_date <= alert_date <= end_date and valid_conf:
        row = (x, y, alert_date.year, alert_date.timetuple().tm_yday, conf)
        return format_row(row, params['format'])
    else:
        return False


def format_row(row, response_format):

     if response_format == 'csv':
         return row
     else:
         return {"year": row[2], "long": row[0], "lat": row[1], "julian_day": row[3], "confidence": row[4]}


def empty_generator():
    # https://stackoverflow.com/a/13243870/
    return
    yield
