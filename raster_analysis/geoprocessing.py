import logging

import numpy as np
import pandas as pd
import json

from raster_analysis.geodesy import get_area
from raster_analysis.io import mask_geom_on_raster, read_windows_parallel, RasterWindow
from raster_analysis.exceptions import RasterReadException

from shapely.geometry import mapping
from aws_xray_sdk.core import xray_recorder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AREA_FIELD = "area"
COUNT_FIELD = "count"
SUM_FIELD = "sum"
FILTERED_AREA_FIELD = "filtered_area"
LAYER_AREA_FIELD = "{raster_id}_area"
FILTER_FIELD = "filter"


def analysis(
    geom,
    analyses,
    analysis_raster_id=None,
    contextual_raster_ids=[],
    aggregate_raster_ids=[],
    start=None,
    end=None,
    extent_year=None,
    threshold=None,
):
    """
    Supported analysis:
        area: calculates geodesic area for unique pixel combinations across all input rasters
        sum: Sums values for all floating point rasters for unique pixel combintions of non-floating point rasters
        count: Counts occurrences of unique pixel combinations across all input rasters

    If threshold is > 0, the 2nd and 3rd raster need to be tcd2000 and tcd2010.
        It will use tcd2000 layer as additional mask and add
    """

    # always log parameters so we can reproduce later
    _log_request(
        geom,
        analyses,
        analysis_raster_id,
        contextual_raster_ids,
        aggregate_raster_ids,
        extent_year,
        threshold,
        start,
        end,
    )

    mean_area = get_area(geom.centroid.y) / 10000
    extent_layer_id = f"tcd_{extent_year}" if extent_year and threshold else None

    raster_windows = _get_raster_windows(
        geom,
        analysis_raster_id,
        contextual_raster_ids,
        aggregate_raster_ids,
        extent_layer_id,
        threshold,
    )

    mask = _get_mask(
        geom, raster_windows, analysis_raster_id, extent_layer_id, start, end
    )

    # apply initial analysis, grouping by all but aggregate fields (these may be later further grouped)
    analysis_groupby_fields = get_raster_id_array(
        analysis_raster_id, contextual_raster_ids
    )

    analysis_result = _analysis(
        analyses,
        raster_windows,
        analysis_groupby_fields,
        aggregate_raster_ids,
        mean_area,
        mask,
    )

    result = analysis_result.to_dict()
    logger.debug("Successfully ran analysis=" + str(analyses))
    logger.info("Ran analysis with result: " + json.dumps(result))

    return result


def _get_detailed_table(
    analysis_result, no_data_value, analysis_raster_id, extent_layer_id
):
    # filter out rows that don't pass filter or where analysis layer is NoData
    filtered_result = analysis_result[
        analysis_result[analysis_raster_id] != no_data_value
    ]

    # if filtering with extent layer, filtered out rows outside the extent
    if extent_layer_id:
        filtered_result = filtered_result[extent_layer_id].drop(
            columns=[extent_layer_id]
        )

    return filtered_result.reset_index(drop=True)


def _get_summary_table(
    analyses,
    analysis_result,
    no_data_value,
    analysis_raster_id,
    contextual_raster_ids,
    aggregate_raster_ids,
    extent_layer_id,
    extent_year,
):
    summary_table = analysis_result.copy()

    if "area" in analyses:
        if extent_layer_id:
            extent_area_name = f"extent_{extent_year}__ha"
            # create new column that just copies area column, but sets any row with filter=false to 0
            summary_table[extent_area_name] = (
                summary_table[AREA_FIELD] * summary_table[extent_layer_id]
            )

            # create new column that copies filtered column, but sets any row where analysis layer is NoData to 0
            layer_area_field = f"{analysis_raster_id}_area__ha"
            summary_table[layer_area_field] = summary_table[extent_area_name] * (
                summary_table[analysis_raster_id] != no_data_value
            )

    drop_columns = [analysis_raster_id]
    if contextual_raster_ids:
        # aggregate by combos of contextual layer values, and drop fields we don't want in the final result
        return (
            summary_table.groupby(contextual_raster_ids)
            .sum()
            .reset_index()
            .drop(columns=drop_columns)
        )
    else:
        # sum and then transpose because Pandas flattens to a series if sum without a groupby
        return pd.DataFrame(summary_table.sum()).transpose().drop(columns=drop_columns)


@xray_recorder.capture("Get Mask")
def _get_mask(geom, raster_windows, analysis_raster_id, extent_layer_id, start, end):
    sample_layer, shifted_affine, no_data = list(raster_windows.values())[0]

    # start by masking geometry onto analysis layer (where mask=True indicates the geom intersects)
    mask = mask_geom_on_raster(sample_layer, shifted_affine, geom)
    # then mask the time interval
    # TODO: is this actually worth the perf bump vs memory allocation? can also just filter at the end
    if analysis_raster_id:
        mask *= _mask_by_nodata(
            raster_windows[analysis_raster_id].data,
            raster_windows[analysis_raster_id].no_data,
        )
        if start or end:
            mask *= _mask_interval(raster_windows[analysis_raster_id].data, start, end)

    if extent_layer_id:
        mask *= raster_windows[extent_layer_id].data

    return mask


def _get_raster_windows(
    geom,
    analysis_raster_id,
    contextual_raster_ids,
    aggregate_raster_ids,
    extent_layer_id,
    threshold,
):
    unique_raster_sources = get_raster_id_array(
        analysis_raster_id,
        contextual_raster_ids,
        aggregate_raster_ids,
        extent_layer_id,
        unique=True,
    )

    raster_windows = read_windows_parallel(unique_raster_sources, geom)

    # fail if analysis layer is empty
    if analysis_raster_id and raster_windows[analysis_raster_id].data.size == 0:
        raise RasterReadException(
            "Analysis raster `" + analysis_raster_id + "` returned empty array"
        )

    # other layers we'll just use raster with all values == 0
    for raster_id in get_raster_id_array(
        contextual_raster_ids, aggregate_raster_ids, extent_layer_id
    ):
        if raster_windows[raster_id].data.size == 0:
            raster_windows[raster_id] = RasterWindow(
                np.zeros(raster_windows[analysis_raster_id].data.shape, dtype=np.uint8),
                raster_windows[analysis_raster_id].shifted_affine,
                raster_windows[raster_id].no_data,
            )

    if extent_layer_id:
        raster_windows[extent_layer_id] = _mask_by_threshold(
            raster_windows[extent_layer_id].data, threshold
        )

    return raster_windows


@xray_recorder.capture("Analysis")
def _analysis(
    analyses,
    raster_windows,
    reporting_raster_ids,
    aggregate_raster_ids,
    mean_area,
    mask,
):
    reporting_columns = [
        np.ravel(raster_windows[raster_id].data) for raster_id in reporting_raster_ids
    ]
    column_maxes = [col.max() + 1 for col in reporting_columns]

    linear_indices = _get_linear_indices(reporting_columns, column_maxes, mask)

    unique_values, counts = _unique(linear_indices)
    unique_value_combinations = np.unravel_index(unique_values, column_maxes)

    result_dict = dict(zip(reporting_raster_ids, unique_value_combinations))

    if "count" in analyses:
        result_dict[COUNT_FIELD] = counts

    if "area" in analyses:
        result_dict[AREA_FIELD] = counts * mean_area

    if aggregate_raster_ids:
        linear_indices = get_inverse(linear_indices, unique_values)

        for raster_id in aggregate_raster_ids:
            result_dict[raster_id] = _get_sum(
                linear_indices, mask, raster_windows[raster_id].data, counts.size
            )

    return pd.DataFrame(result_dict)


@xray_recorder.capture("Get Linear Indicies")
def _get_linear_indices(columns, dims, mask):
    return np.compress(
        np.ravel(mask), np.ravel_multi_index(columns, dims).astype(np.uint32)
    )


@xray_recorder.capture("Get Sum")
def _get_sum(linear_indices, mask, data, bins):
    return np.bincount(linear_indices, weights=_extract(mask, data), minlength=bins)


@xray_recorder.capture("Unique")
def _unique(array):
    return np.unique(array, return_counts=True)


@xray_recorder.capture("Get Inverse")
def get_inverse(linear_indices, unique_values):
    return np.digitize(linear_indices, unique_values, right=True)


@xray_recorder.capture("Create Filter")
def _get_filter(filter_raster, filter_intervals):
    filter = np.zeros(filter_raster.shape, dtype=np.bool)

    for interval in filter_intervals:
        filter += (filter_raster > interval[0]) * (filter_raster <= interval[1])

    logger.debug("Successfully create filter")
    return filter


@xray_recorder.capture("NumPy Extract")
def _extract(mask, data):
    return np.extract(mask, data)


@xray_recorder.capture("Mask by Threshold")
def _mask_by_threshold(raster, threshold):
    return raster > threshold


@xray_recorder.capture("Mask by NoData")
def _mask_by_nodata(raster, no_data):
    return raster != no_data


@xray_recorder.capture("Mask Interval")
def _mask_interval(raster, start, end):
    if start and end:
        return (raster >= start) * (raster <= end)
    elif start:
        return raster >= start
    elif end:
        return raster <= end


def get_raster_id_array(*raster_id_args, unique=False):
    ids = []

    for raster_id_arg in raster_id_args:
        if raster_id_arg:
            if isinstance(raster_id_arg, list):
                ids += raster_id_arg
            elif isinstance(raster_id_arg, str):
                ids.append(raster_id_arg)
            else:
                raise ValueError(
                    "Cannot add raster id of type"
                    + str(type(raster_id_arg))
                    + "to raster id array."
                )

    return list(set(ids)) if unique else ids


def _log_request(
    geom,
    analyses,
    analysis_raster_id,
    contextual_raster_ids,
    aggregate_raster_ids,
    extent_year,
    threshold,
    start,
    end,
):
    logger.info(
        f"Running analysis with parameters: "
        + f"\nanalyses: {analyses}"
        + f"\nanalysis_raster_id: {analysis_raster_id}"
        + f"\ncontextual_raster_ids: {contextual_raster_ids}"
        + f"\naggregate_raster_ids: {aggregate_raster_ids}"
        + f"\nextent_year: {extent_year}"
        + f"\nthreshold: {threshold}"
        + f"\nstart: {start}"
        + f"\nend: {end}"
        + f"\ngeom: {json.dumps(mapping(geom))}"
    )
