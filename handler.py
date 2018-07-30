import json
import os
import sys


# add path to included packages
# these are all stored in the root of the zipped deployment package
root_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, root_dir)

from geop import geo_utils, geoprocessing
from serializers import gfw_api
from utilities import util 

data_dir = os.path.join(root_dir, 'data')
glad_raster = os.path.join(data_dir, 'glad.vrt')


def glad_alerts(event, context, ras=glad_raster):

    geom, area_ha = util.get_shapely_geom(event)

    try:
        params = util.validate_glad_params(event)
    except ValueError, e:
        return gfw_api.api_error(str(e))

    stats = geoprocessing.count(geom, ras)

    hist = util.unpack_glad_histogram(stats, params)

    return gfw_api.serialize_glad(hist, area_ha, params['aggregate_by'], params['period'])


if __name__ == '__main__':

    aoi = {"features":[{"properties":{},"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[140.2137,-6.3999],[140.3078,-6.3685],[140.3249,-6.3924],[140.3249,-6.4606],[140.3064,-6.49],[140.2274,-6.4838],[140.2068,-6.434],[140.2137,-6.3999]]]}}],"crs":{},"type":"FeatureCollection"}

    # why this crazy structure? Oh lambda . . . sometimes I wonder
    event = {
        'body': json.dumps({'geojson': aoi}),
        'queryStringParameters': {'aggregate_values': False, 'analysis': 'extent'}
    }


    print glad_alerts(event, None)

