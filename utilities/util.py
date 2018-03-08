import json
import datetime
from collections import defaultdict
import sys
from functools import partial

import pyproj
from shapely.geometry import shape
from shapely.ops import transform


def unpack_count_histogram(analysis_type, stats):

    value_offset = 0

    if analysis_type == 'loss':
        value_offset = 2000

    output_dict = {}

    for key, pixel_count in stats.iteritems():
        ras1, area = key.split('::')

        ras1 = value_offset + int(float(ras1))
        area_ha = float(area) * float(pixel_count) / 10000.

        try:
            output_dict[ras1] += area_ha
        except KeyError:
            output_dict[ras1] = area_ha

    return output_dict


def get_shapely_geom(event):

    print event
    geojson = json.loads(event['body'])['geojson']

    if len(geojson['features']) > 1:
        raise ValueError('Currently accepting only 1 feature at a time')

    # grab the actual geometry-- that's the level on which shapely operates
    geom = shape(geojson['features'][0]['geometry'])
    area_ha = get_polygon_area(geom)

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
                 'day': grouped_and_to_rows([(x.year, x.strftime('%Y-%m-%d')) for x in k], v, 'day')
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
        print conf_days
        total_days = int(conf_days[1:])
        year = total_days / 365 + 2015
        julian_day = total_days % 365

        # https://stackoverflow.com/questions/17216169/
        alert_date = datetime.datetime(year, 1, 1) + datetime.timedelta(julian_day - 1)

        try:
    	    date_dict[alert_date] += count # if date exists, add count to id
        except KeyError:
    	    date_dict[alert_date] = count # if date doesn't exist, create it, set equal to count

        # filter dates by period
        start_date, end_date = period_to_dates(period)
        filtered_by_period = {alert_date : count for alert_date, count in date_dict.iteritems() if start_date < alert_date < end_date}

        resp_dict = create_resp_dict(filtered_by_period)

    return resp_dict.get(hist_type, {'all': resp_dict})


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

        row['count'] = val
        final_list.append(row)

    return final_list

def set_default_period():
    today = datetime.datetime.now().date()
    today = today.strftime('%Y-%m-%d')

    return "2015-01-01,{}".format(today)

def validate_glad_params(event):

    params = event.get('queryStringParameters')

    if not params:
        params = {}

    # if period isn't supplied, set default
    period = params.get('period', set_default_period())

    params['period'] = period

    # make sure the supplied period is in the right format, dates make sense
    try:
        check_dates(period)
    except ValueError, e:
        raise ValueError(e)


    agg_values = params.get('aggregate_values', False)
    agg_by = params.get('aggregate_by')

    if agg_values in ['true', 'TRUE', 'True', True]:
        agg_values = True

    agg_list = ['day', 'week', 'month', 'year', 'quarter', 'all']

    if agg_by not in agg_list or agg_values != True:
        msg = 'For this batch service, aggregate_values must be True, and ' \
              'aggregate_by must be in {}'.format(', '.join(agg_list))
        raise ValueError(msg)


    return params


def check_dates(period):

    try:
        start_date, end_date = period_to_dates(period)
    except ValueError:
        raise ValueError('period must be formatted as YYYY-mm-dd,YYYY-mm-dd')

    if start_date > end_date:
        raise ValueError('Start date must be <= end date')

    earliest_date = datetime.datetime.strptime('2015-01-01', '%Y-%m-%d')
    if start_date < earliest_date:
        raise ValueError('Start date must be later than Jan 1, 2015')


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
