import numpy as np
from raster_analysis.utilities.grid import get_tile_id, get_raster_url
from raster_analysis.utilities.arrays import to_structured_array

# import pandas as pd
import json
from raster_analysis.utilities.io import read_window, mask_geom_on_raster, read_window_ignore_missing
from raster_analysis.utilities.geodesy import get_area


def sum_analysis(geom, *raster_ids, threshold=0, area=True):
    """
    If area is true, it will always sum the area for unique layer combinations.
    If area is false, it will sum values of the last input raster
    If threshold is > 0, the 2nd and 3rd raster need to be tcd2000 and tcd2010
    :param geom:
    :param rasters:
    :param threshold:
    :param area:
    :return:
    """

    mean_area = get_area((geom.bounds[3] - geom.bounds[1]) / 2)
    tile_id = get_tile_id(geom)

    raster = get_raster_url(raster_ids[0], tile_id)
    masked_data, no_data, _ = mask_geom_on_raster(geom, raster)

    if masked_data.any():
        if threshold > 0:
            tcd_2000_url = get_raster_url(raster_ids[1], tile_id)
            tcd_2000_mask = _mask_by_threshold(read_window(tcd_2000_url, geom)[0], threshold)

            tcd_2010_url = get_raster_url(raster_ids[2], tile_id)
            tcd_2010_mask = _mask_by_threshold(read_window(tcd_2010_url, geom)[0], threshold)

            tcd_2000_extent = tcd_2000_mask * masked_data.mask * mean_area / 10000
            tcd_2010_extent = tcd_2010_mask * masked_data.mask * mean_area / 10000

            rasters_to_process = raster_ids[3:]

        else:
            tcd_2000_mask = 1
            rasters_to_process = raster_ids[1:]
            tcd_2000_extent = None
            tcd_2010_extent = None

        value_mask = _mask_by_nodata(masked_data.data, no_data)
        final_mask = value_mask * tcd_2000_mask * masked_data.mask

        primary_array = to_structured_array(masked_data.data, raster_ids[0])

        contextual_array = _build_array(final_mask, primary_array, *rasters_to_process, geom=geom)

        result = {"data": _sum_area(contextual_array, mean_area).tolist()}
        result["extent_2000"] = tcd_2000_extent
        result["extent_2010"] = tcd_2010_extent
        return result
    else:
        return {}


def _mask_by_threshold(raster, threshold):
    return raster > threshold


def _mask_by_nodata(raster, no_data):
    return raster != no_data


def _build_array(mask, array, *raster_ids, geom=None):

    tile_id = _get_tile_id(geom)

    result = np.extract(mask, array)

    for raster_id in raster_ids:
        raster = _get_raster_url(raster_id, tile_id)
        data, _, _ = read_window_ignore_missing(raster, geom)
        if data.any():
            values = np.extract(mask, data)
        else:
            values = np.zeros(len(result))
        data = None
        result = np.dstack((result, values))

    return result[0]


def _sum_area(array, area):
    unique_rows, occur_count = np.unique(array, axis=0, return_counts=True)
    total_area = (occur_count * area).reshape(len(occur_count),1)

    return np.hstack((unique_rows, total_area))


def _count(array):
    unique_rows, occur_count = np.unique(array, axis=0, return_counts=True)
    counts = occur_count.reshape(len(occur_count), 1)

    return (np.hstack((unique_rows, counts))).tolist()


def _sum(array):
    row_length = len(array[0])
    row_number = len(array)
    group = np.resize(array, row_length-1, row_number)
    values = np.array_split(array,len(array[0]),1)[len(array[0])-1]
