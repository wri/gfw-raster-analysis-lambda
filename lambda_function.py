from raster_analysis import geoprocessing
from raster_analysis.utilities import grid_id
from shapely.geometry import shape


BASE_URL = "/vsis3/gfw-files/2018_update/{raster_id}/{tile_id}.tif"


def lambda_handler(event, context):

    raster_ids = event.raster_ids
    analysis = event.analsis_id
    geometry = shape(event.geometry)
    threshold = event.threshold

    centroid = geometry.centroid

    tile_id = grid_id.get_gridid(centroid)

    rasters = list()
    for raster_id in raster_ids:
        raster = BASE_URL.format(raster_id=raster_id, tile_id=tile_id)
        rasters.append(raster)

    if analysis == "area_sum":
        result = geoprocessing.sum_analysis(geoprocessing, rasters, threshold=threshold, area=True)
    else:
        result = {"error": "anlysis type unknown"}

    return result
