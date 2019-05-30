import boto3
from shapely.geometry import Polygon, MultiPolygon
import json
from urlparse import urlparse


def check_extent(user_poly, raster):
    s3 = boto3.resource('s3')

    bucket = s3.Bucket('palm-risk-poc')

    # read context of the index file
    parsed = urlparse(raster)
    s3obj = parsed.path.replace('data.vrt', 'index.geojson')[1:]
    d = bucket.Object(s3obj).get()['Body'].read()

    # get contents from string to dictionary
    d = json.loads(d)

    # get index geom
    poly_list = [Polygon(x['geometry']['coordinates'][0]) for x in d['features']]
    
    # read in indexgeom as shapely multipolygon
    index_geom = MultiPolygon(poly_list)

    # get user geom
    user_geom = user_poly['features'][0]['geometry']['coordinates'][0]
    
    # read in user geom as shapely geometry
    user_poly = Polygon(user_geom)

    # check if polygons intersect
    return user_poly.intersects(index_geom)
    