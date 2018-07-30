import json
import os
import sys

from flask import Flask, jsonify
app = Flask(__name__)


# add path to included packages
# these are all stored in the root of the zipped deployment package
root_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, root_dir)

from geop import geo_utils, geoprocessing
from serializers import gfw_api
from utilities import util 


@app.route("/glad-alerts", methods=['POST'])
def glad_alerts():

    geom, area_ha = util.get_shapely_geom()

    try:
        params = util.validate_glad_params()
    except ValueError, e:
        return gfw_api.api_error(str(e))

    if os.environ.get('ENV') == 'test':
        glad_raster = os.path.join(root_dir, 'test', 'data', 'afr_all_years_clip.tif')
    else:
        glad_raster = os.path.join(root_dir, 'data', 'glad.vrt')

    stats = geoprocessing.count(geom, glad_raster)

    hist = util.unpack_glad_histogram(stats, params)

    return gfw_api.serialize_glad(hist, area_ha, params['aggregate_by'], params['period'])


@app.route("/glad-alerts/download", methods=['POST'])
def download_glad():
    return jsonify({'this': 'ok'}), 200


