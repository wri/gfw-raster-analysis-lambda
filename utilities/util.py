import json
import datetime
from collections import defaultdict

from shapely.geometry import shape


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
    return shape(geojson['features'][0]['geometry'])


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


def unpack_glad_histogram(stats, hist_type):

	date_dict = {}

	for conf_days, count in stats.iteritems():
	    total_days = int(conf_days[1:])
	    year = total_days / 365 + 2015
	    julian_day = total_days % 365

	    # https://stackoverflow.com/questions/17216169/
	    alert_date = datetime.datetime(year, 1, 1) + datetime.timedelta(julian_day - 1)

	    try:
		    date_dict[alert_date] += count # if date exists, add count to id
	    except KeyError:
		    date_dict[alert_date] = count # if date doesn't exist, create it, set equal to count

        resp_dict = create_resp_dict(date_dict)

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

