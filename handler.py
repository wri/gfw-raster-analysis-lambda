import json
import requests
import pyproj
import subprocess

from shapely.geometry import mapping, Point, shape
from shapely.ops import cascaded_union
from geop import geo_utils, geoprocessing


def mill(event, context):
    lat = event['queryStringParameters']['lat']
    lon = event['queryStringParameters']['lon']

    wgs84_str = 'EPSG:4326'
    eckVI_str = "esri:54010"

    eckVI_proj = pyproj.Proj(init="{}".format(eckVI_str))
    eck_point = Point(eckVI_proj(lon, lat))

    buffered = eck_point.buffer(50000)

    output = geo_utils.reproject(buffered, eckVI_str, wgs84_str)
    geojson_output = '/tmp/mill_50km_buffer.geojson'

    with open(geojson_output, 'w') as outfile:
        json.dump(mapping(output), outfile)

    layer_dict = {'gadm28': 'gadm28_adm0.shp',
                  'wdpa': 'wdpa_protected_areas.shp',
                  'peat': 'idn_peat_lands.shp',
                  'primary_forest': 'idn_primary_forest_shp.shp'}

    mill_dict = {}

    for layer_id, layer_shp in layer_dict.iteritems():

        mill_dict[layer_id] = {}

        print 'starting AOI intersect for {}'.format(layer_id)
        aoi_intersect = clip_layer_by_mill(geojson_output, layer_shp)

        # project to eckVI to get proper area
        aoi_intersect_projected = geo_utils.reproject(aoi_intersect, wgs84_str, eckVI_str)
        area_overlap_ha = aoi_intersect_projected.area / 10000.0
        print 'area of overlap: {} ha'.format(area_overlap_ha)

        mill_dict[layer_id]['area_ha'] = area_overlap_ha

        print 'calculating stats against 2000 - 2016 loss'
        loss_dict = calc_loss(aoi_intersect)

        mill_dict[layer_id]['loss'] = loss_dict



    # loop over various layers of interest
    # land, peat, wdpa, etc
    # intersect with each one, project to eckVI to tabulate area
    # then count loss pixels


def clip_layer_by_mill(mill_buffer_geojson, layer_shp):

    s3_path = 'palm-risk-poc/data'
    layer_path = '/vsicurl/http://s3.amazonaws.com/{}/{}'.format(s3_path, layer_shp)

    layer_name = layer_shp.replace('.shp', '')

    local_intersect = 'intersect.geojson'

    cmd = ['ogr2ogr', '-f', 'GeoJSON', local_intersect, layer_path,
           '-clipsrc', mill_buffer_geojson]
    subprocess.check_call(cmd)

    print ' '.join(cmd)

    with open(local_intersect) as infile:
        geojson = json.load(infile)

    geoms = [shape(f['geometry']) for f in geojson['features']]

    return cascaded_union(geoms)


def calc_loss(aoi_intersect):

    loss_vrt = r's3://gfw2-data/forest_change/hansen_2016_masked_30tcd/data.vrt'

    raster_dict = geoprocessing.count(aoi_intersect, loss_vrt)

    print raster_dict

    return raster_dict


if __name__ == '__main__':
    d = {'queryStringParameters': {'lon': 112.8515625, 'lat': -2.19672724}}
    # d = {'lon': 112.8515625, 'lat': -2.19672724}

    mill(d, None)
    # raster_stats(d, None)
