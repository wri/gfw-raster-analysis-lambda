import numpy as np
import pandas as pd
import json
from raster_analysis.utilities.geo_utils import read_window, mask_geom_on_raster, get_area, read_window_ignore_missing


def sum_analysis(geom, *rasters, threshold=0, area=True):
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
    masked_data, no_data, _ = mask_geom_on_raster(geom, rasters[0])
    mean_area = get_area((geom.bounds[3] - geom.bounds[1]) / 2)

    if masked_data.any():
        if threshold > 0:
            tcd_2000_mask = _mask_by_threshold(read_window(rasters[1], geom)[0], threshold)
            tcd_2010_mask = _mask_by_threshold(read_window(rasters[2], geom)[0], threshold)

            tcd_2000_extent = tcd_2000_mask * masked_data.mask * mean_area/10000
            tcd_2010_extent = tcd_2010_mask * masked_data.mask * mean_area/10000

            rasters_to_process = rasters[3:]

        else:
            tcd_2000_mask = 1
            rasters_to_process = rasters[1:]
            tcd_2000_extent = None
            tcd_2010_extent = None

        value_mask = _mask_by_nodata(masked_data.data, no_data)
        final_mask = value_mask * tcd_2000_mask * masked_data.mask

        contextual_array = _build_array(final_mask, masked_data.data, *rasters_to_process, geom=geom, area=area)

        result = json.loads(_sum(contextual_array).to_json())
        result["extent_2000"] = tcd_2000_extent
        result["extent_2010"] = tcd_2010_extent
        return result
    else:
        return json.loads(pd.DataFrame().to_json())


def _mask_by_threshold(raster, threshold):
    return raster > threshold


def _mask_by_nodata(raster, no_data):
    return raster != no_data


def _build_array(mask, array, *rasters, geom=None, area=False):

    result = np.extract(mask, array)

    for raster in rasters:
        data, _, _ = read_window_ignore_missing(raster, geom)
        if data.any():
            values = np.extract(mask, data)
        else:
            values = np.zeros(len(result))
        data = None
        result = np.dstack((result, values))

    if area:
        area = get_area((geom.bounds[3] - geom.bounds[1]) / 2)
        result = np.dstack((result, np.ones(len(result[0])) * area))

    return result[0]


def _sum(array):

    df = pd.DataFrame(array)
    df.columns = ["col{}".format(i) if i < len(df.columns) - 1 else "value" for i in df.columns]
    result = df.groupby(list(df.columns[:-1]), axis=0).sum().reset_index()
    return result


