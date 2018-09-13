import json
import os
import sys

from flask import Flask, jsonify, request, Response
app = Flask(__name__)
app.url_map.strict_slashes = False


# add path to included packages
# these are all stored in the root of the zipped deployment package
root_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, root_dir)

from geop import geo_utils, geoprocessing
from serializers import gfw_api
from utilities import util, errors 


@app.route("/glad-alerts", methods=['POST'])
@app.route("/glad-alerts/download", methods=['POST'])
def glad_alerts():

    geom, area_ha = util.get_shapely_geom()
    params = util.validate_glad_params()

    if os.environ.get('ENV') == 'test':
        glad_raster = os.path.join(root_dir, 'test', 'data', 'afr_all_years_clip.tif')
    else:
        glad_raster = os.path.join(root_dir, 'data', 'glad.vrt')

    if 'download' in request.url_rule.rule:
        return download(geom, glad_raster, params)

    else:
        return stats(geom, glad_raster, params, area_ha)


def stats(geom, glad_raster, params, area_ha):

    stats = geoprocessing.count(geom, glad_raster)

    hist = util.unpack_glad_histogram(stats, params)

    return gfw_api.serialize_glad(hist, area_ha, params['aggregate_by'], params['period'])


def download(geom, glad_raster, params):

    masked_data, shifted_affine = geo_utils.mask_geom_on_raster(geom, glad_raster)

    mimetype_dict = {'csv': 'text/csv', 'json': 'application/json'}

    # make sure that our AOI covers the raster of interest
    if masked_data.any():
        rows = [util.filter_rows(row, params) for row in geo_utils.array_to_xyz_rows(masked_data, shifted_affine)]
        rows = filter(lambda x: x is not False, rows)
    else:
        rows = []

    if params['format'] == 'csv':
         rows = ['longitude,latitude,year,julian_day,confidence\n'] + rows
    else:
         rows = json.dumps({'data': rows})

    return Response(rows, mimetype=mimetype_dict[params['format']])


@app.errorhandler(errors.Error)
def handle_error(error):
    return error.serialize

