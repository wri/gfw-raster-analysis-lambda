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
            data = to_structured_array(data, raster_id)
            values = np.extract(mask, data)
        else:
            values = to_structured_array(np.zeros(len(arrays[0])), raster_id, 'bool_')

        print(raster_id)
        print(values)

        return values

    tile_id = get_tile_id(geom)

    arrays = [np.extract(mask, array)]

    for raster_id in raster_ids:
        values = _get_values()
        arrays.append(values)

    result = concat_arrays(*arrays)

    return result


def concat_arrays(*arrays):
    dts = list()
    for array in arrays:
        dts += _dtype_to_list(array.dtype)
    dt = np.dtype(dts)

    array = np.empty(len(arrays[0]), dtype=dt)

    return _fill_array(array, *arrays)


def fields_view(arr, fields):
    # https://stackoverflow.com/questions/15182381/how-to-return-a-view-of-several-columns-in-numpy-structured-array
    dtype2 = np.dtype({name: arr.dtype.fields[name] for name in fields})
    return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)


def get_fields_by_type(dtypes, dtype, exclude=False):
    if exclude:
        return [n for n in dtypes.names if not np.issubdtype(dtypes[n], dtype)]
    else:
        return [n for n in dtypes.names if np.issubdtype(dtypes[n], dtype)]


def _dtype_to_list(dtype):
    return [(n, dtype[n]) for n in dtype.names]


def _fill_array(fill_array, *arrays):
    for array in arrays:
        for n in array.dtype.names:
            fill_array[n] = array[n]

    return fill_array
