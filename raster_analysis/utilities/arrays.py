from raster_analysis.utilities.grid import get_tile_id, get_raster_url
from raster_analysis.utilities.io import read_window_ignore_missing
import numpy as np


def to_structured_array(array, name, dt=None):
    if not dt:
        dt = array.dtype
    array.dtype = np.dtype([(name, dt)])
    return array


def build_array(mask, array, *raster_ids, geom=None):

    def _get_values():
        raster = get_raster_url(raster_id, tile_id)
        data, _, _ = read_window_ignore_missing(raster, geom)
        if data.any():
            data = to_structured_array(array, raster_id)
            values = np.extract(mask, data)
        else:
            values = to_structured_array(np.zeros(len(result)), 'bool_')

        return values

    tile_id = get_tile_id(geom)

    result = np.extract(mask, array)

    for raster_id in raster_ids:
        values = _get_values()
        result = _build_array(result, values)

    return result


def _dtype_to_list(dt):
    return [(n, dt[n]) for n in dt.names]


def _fill_array(fill_array, *arrays):
    for array in arrays:
        for n in array.dtype.names:
            fill_array[n] = array[n]

    return fill_array


def _build_array(a, b):
    dt = np.dtype(_dtype_to_list(a.dtype) + _dtype_to_list(b.dtype))
    array = np.empty(len(a), dtype=dt)

    return _fill_array(array, a, b)
