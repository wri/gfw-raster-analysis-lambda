import numpy as np
import pandas as pd
import rasterio
from utilities.errors import Error
import json
from geo_utils import mask_geom_on_raster, get_area


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
    masked_data, no_data = mask_geom_on_raster(geom, rasters[0])
    if masked_data.any():
        if threshold > 0:
            tcd_2000_mask = _mask_by_threshold(_read_window(rasters[1], geom.extent), threshold)
            tcd_2010_mask = _mask_by_threshold(_read_window(rasters[2], geom.extent), threshold)

            mean_area = get_area((geom.extent.maxy - geom.extent.miny) / 2)
            tcd_2000_extent = tcd_2000_mask * masked_data.mask * mean_area/10000
            tcd_2010_extent = tcd_2010_mask * masked_data.mask * mean_area/10000

            rasters_to_process = [rasters[0]]
            if len(rasters)>3:
                rasters_to_process += rasters[3:]

        else:
            tcd_2000_mask = 1
            rasters_to_process = rasters
            tcd_2000_extent = None
            tcd_2010_extent = None

        value_mask = _mask_by_nodata(masked_data, no_data)
        final_mask = value_mask * tcd_2000_mask * masked_data.mask

        contextual_array = _build_array(final_mask, rasters_to_process, extent=geom.extent, area=area)

        j = json.loads(_aggregate(contextual_array))
        j["extent_2000"] = tcd_2000_extent
        j["extent_2010"] = tcd_2010_extent
        return j
    else:
        return pd.DataFrame().to_json()


def _read_window(raster, window):
    with rasterio.Env():
        with rasterio.open(raster) as src:
            try:
                data = src.read(1, masked=True, window=window)
            except MemoryError:
                raise Error('Out of memory- input polygon or input extent too large. '
                            'Try splitting the polygon into multiple requests.')
    return data


def _mask_by_threshold(raster, threshold):
    return raster > threshold


def _mask_by_nodata(raster, no_data):
    return raster != no_data


def _build_array(final_mask, *rasters, extent, area=True):
    result = np.array()
    for raster in rasters:
        values = np.extract(final_mask, raster)
        result = np.ma.dstack(result, values)

    if area:
        area = get_area((extent.maxy - extent.miny) / 2)
        result = np.ma.dstack(result,np.ones(result.len) * area)

    return result


def _aggregate(array, extent):

    df = pd.DataFrame(array)
    df.columns = ["col{}".format(i) if i < len(df.columns) - 1 else "value" for i in df.columns]

    return df.groupby(list(df.columns[:-1]), axis=0).sum().to_json


