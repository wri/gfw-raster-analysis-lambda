import json
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

    print output_dict

    return output_dict


def get_shapely_geom(event):

    print event
    geojson = json.loads(event['body'])['geojson']

    if len(geojson['features']) > 1:
        raise ValueError('Currently accepting only 1 feature at a time')

    # grab the actual geometry-- that's the level on which shapely operates
    return shape(geojson['features'][0]['geometry'])
