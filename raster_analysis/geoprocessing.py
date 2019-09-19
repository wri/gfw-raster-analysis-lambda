import numpy as np
from raster_analysis.grid import get_tile_id, get_raster_url
from raster_analysis.arrays import (
    to_structured_array,
    build_array,
    concat_arrays,
    fields_view,
    get_fields_by_type,
    fill_array,
    dtype_to_list,
)
from raster_analysis.io import read_window, mask_geom_on_raster
from raster_analysis.geodesy import get_area
from collections import namedtuple

Filter = namedtuple("Filter", "raster_id threshold")


def analysis(geom, *raster_ids, filters=None, analysis="area"):
    """
    Supported analysis:
        area: calculates geodesic area for unique pixel combinations across all input rasters
        sum: Sums values for all floating point rasters for unique pixel combintions of non-floating point rasters
        count: Counts occurrences of unique pixel combinations across all input rasters

    If threshold is > 0, the 2nd and 3rd raster need to be tcd2000 and tcd2010.
        It will use tcd2000 layer as additional mask and add
    """

    def _analysis():
        if analysis == "area":
            return _sum_area(contextual_array, mean_area)
        elif analysis == "sum":
            return _sum(contextual_array)
        elif analysis == "count":
            return _count(contextual_array)
        else:
            raise ValueError("Unknown analysis: " + analysis)

    result = dict()

    mean_area = get_area((geom.bounds[3] - geom.bounds[1]) / 2) / 10000
    tile_id = get_tile_id(geom)

    raster = get_raster_url(raster_ids[0], tile_id)
    masked_data, no_data, _ = mask_geom_on_raster(geom, raster)

    if masked_data.mask.any():
        value_mask = _mask_by_nodata(masked_data.data, no_data)
        mask = masked_data.mask * value_mask

        if filters:
            cumulative_filter_mask = masked_data.mask

            for curr_filter in filters:
                curr_filter_url = get_raster_url(curr_filter.raster_id, tile_id)
                curr_filter_mask = _mask_by_threshold(
                    read_window(curr_filter_url, geom)[0], curr_filter.threshold
                )

                cumulative_filter_mask *= curr_filter_mask
            mask *= cumulative_filter_mask

        primary_array = to_structured_array(masked_data.data, raster_ids[0])

        contextual_array = build_array(mask, primary_array, *raster_ids[1:], geom=geom)

        a = _analysis()

        result["extent"] = mask.sum() * mean_area
        result["data"] = a.tolist()
        result["dtype"] = dtype_to_list(a.dtype)

        return result

    else:
        return dict()


def _mask_by_threshold(raster, threshold):
    return raster > threshold


def _mask_by_nodata(raster, no_data):
    return raster != no_data


def _sum_area(array, area):
    unique_rows, occur_count = np.unique(array, axis=0, return_counts=True)
    total_area = occur_count * area
    total_area.dtype = np.dtype([("AREA", "float")])

    return concat_arrays(unique_rows, total_area)


def _count(array):
    unique_rows, occur_count = np.unique(array, axis=0, return_counts=True)
    occur_count.dtype = [("COUNT", "int")]

    return concat_arrays(unique_rows, occur_count)


def _sum(array):
    group_fields = get_fields_by_type(array.dtype, "float", exclude=True)
    value_fields = get_fields_by_type(array.dtype, "float", exclude=False)

    group_array = fields_view(array, group_fields)
    value_array = fields_view(array, value_fields)

    unique_rows, occur_count = np.unique(group_array, axis=0, return_counts=True)

    sum_array = np.empty(len(unique_rows), dtype=value_array.dtype)

    for field in value_fields:
        field_sum = list()

        for i in unique_rows:
            mask = group_array == i
            masked_values = np.extract(mask, value_array)

            field_sum.append(masked_values[field].sum())

        sum_array = fill_array(sum_array, np.array(field_sum, dtype=[(field, "float")]))

    return concat_arrays(unique_rows, sum_array)
