from __future__ import print_function
from __future__ import division

import base64
import json
import requests

from collections import defaultdict
from concurrent import futures
from shapely.geometry import mapping

from geop import request_utils as req
from geop import geo_utils, geoprocessing, tiles

s3_root = 's3://simple-raster-processing-files/'
api_root = 'https://mt2qfe33cl.execute-api.us-east-1.amazonaws.com/dev/'

raster_bucket = {
    'nlcd': s3_root + 'nlcd_512.tif',
    'soil': s3_root + 'hydro_soils_512.tif',
    'nlcd_wm': s3_root + 'nlcd_webm_512_ovr.tif',
    'soil_wm': s3_root + 'hydro_soils_webm_512_ovr.tif',
    'ned': s3_root + 'boston_ned.tif'
}

BIG_AREA = 40000000000


def counts(event, context):
    body = json.loads(event['body'])
    config = req.parse_config(body)

    geom = config['query_polygon']
    raster_path = raster_bucket[config['raster_paths'][0]]

    total, count_map = geoprocessing.count(geom, raster_path)
    print(total)
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'total': total,
            'counts': count_map
        })
    }


def counts_parallel(event, context):
    def fn(payload):
        return requests.post(api, data=payload).json()

    body = json.loads(event['body'])
    config = req.parse_config(body)

    geom = config['query_polygon']
    raster_path = raster_bucket[config['raster_paths'][0]]

    api = api_root + 'demo/counts'

    if geom.area > BIG_AREA:
        geoms = geo_utils.subdivide_polygon(geom, 150000)
        print('subdividing to {} via api'.format(len(geoms)))
        total, count_map = accumulate_counts(geoms, event, fn)
    else:
        total, count_map = geoprocessing.count(geom, raster_path)

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'total': total,
            'counts': count_map
        })
    }


def accumulate_counts(geoms, event, fn):
    def update_geom(options, geom):
        new_options = options.copy()
        poly = mapping(geo_utils.reproject(geom,
                                           from_srs='epsg:5070',
                                           to_srs='epsg:4326'))
        body = json.loads(new_options['body'])
        body['queryPolygon'] = poly

        return json.dumps(body)

    sub_requests = [update_geom(event, geom) for geom in geoms]

    # Lambda has a default limit of 100 simultaneous functions
    max_workers = min(len(geoms), 100)
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        total = 0
        counts = defaultdict(int)

        jobs = [executor.submit(fn, request) for request in sub_requests]
        for future in futures.as_completed(jobs):
            res = future.result()
            print(res)
            total += res['total']
            for key, val in res['counts'].iteritems():
                counts[key] += val

        return total, counts


def extract(event, context):
    def fn(payload):
        return requests.post(api, data=payload).json()

    body = json.loads(event['body'])
    config = req.parse_config(body)

    geom = config['query_polygon']
    args = event['pathParameters']
    raster_path = raster_bucket[config['raster_paths'][0]]
    value = int(args.get('id', 11))

    api = api_root + 'dev/extract/{}'.format(value)

    print('using', value)
    if geom.area > BIG_AREA:
        geoms = geo_utils.subdivide_polygon(geom, 150000)
        print('subdividing to {} via api'.format(len(geoms)))
        features = accumulate_features(geoms, event, value, fn)
    else:
        features = geo_utils.as_json(geoprocessing.extract(geom, raster_path,
                                                           value))
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'features': features
        })
    }


def accumulate_features(geoms, event, value, fn):
    def update_geom(options, geom):
        new_options = options.copy()
        poly = mapping(geo_utils.reproject(geom,
                                           from_srs='epsg:5070',
                                           to_srs='epsg:4326'))
        body = json.loads(new_options['body'])
        body['queryPolygon'] = poly

        return json.dumps(body)

    sub_requests = [update_geom(event, geom) for geom in geoms]

    # Lambda has a default limit of 100 simultaneous functions
    max_workers = min(len(geoms), 100)
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        jobs = [executor.submit(fn, request) for request in sub_requests]

        features = []
        for future in futures.as_completed(jobs):
            res = future.result()
            print('features', len(res['features']))
            features += res['features']

        return features


def sample(event, context):
    body = json.loads(event['body'])
    config = req.parse_config(body)
    geom = config['query_line']

    value = geoprocessing.sample_along_line(geom, raster_bucket['ned'])

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'value': value
        })
    }


def tile(event, context):
    args = event['pathParameters']
    layer = args.get('layer', 'nlcd_wm')
    bbox = geo_utils.tile_to_bbox(int(args['z']),
                                  int(args['x']),
                                  int(args['y']))

    data = tiles.render_tile(bbox, raster_bucket[layer])
    img = base64.b64encode(data.getvalue())

    return img_response(img)


def priority(event, context):
    """
    A contrived prioritization analysis to identify areas where Green
    Stormwater Infrastructure projects would have the most benefit. This
    demonstrates chaining a few geoprocessing tasks together:  First, two
    layers are reclassified into normalized priority scores, which are then
    applied to a weighted overlay, determining an overall priority score. This
    final layer is then rendered visually to denote where GSI projects could
    have a high impact.  The reclassifications and weights could easily be
    provided by the client, exposing they dynamic nature of this on-the-fly
    processing.
    """
    # This would need to otherwise be specified in a config.
    # Requirements are EPSG:3857
    nlcd = raster_bucket['nlcd_wm']
    soil = raster_bucket['soil_wm']

    args = event['pathParameters']
    urban = int(args.get('urban', 10))
    forest = int(args.get('forest', 1))
    bbox = geo_utils.tile_to_bbox(int(args['z']),
                                  int(args['x']),
                                  int(args['y']))

    data = tiles.weighted_overlay_tile(bbox, urban, forest, nlcd, soil)
    img = base64.b64encode(data.getvalue())

    return img_response(img)


def img_response(img):
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'image/png'
        },
        'body': img,
        'isBase64Encoded': True
    }
