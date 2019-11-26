import logging

import numpy as np
import numpy.ma as ma
import pandas as pd
import json
import gc
import math

from raster_analysis.geodesy import get_area
from raster_analysis.io import mask_geom_on_raster, read_windows_parallel, RasterWindow
from raster_analysis.exceptions import RasterReadException

from shapely.geometry import mapping
from aws_xray_sdk.core import xray_recorder

import resource

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
    analysis_raster_id,
    contextual_raster_ids=[],
    aggregate_raster_ids=[],
    filter_raster_id=None,
    filter_intervals=None,
    density_raster_ids=[],
    analyses=["count", "area"],
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
    logger.info(
        "Running analysis with parameters: "
        + "analyses: "
        + str(analyses)
        + ", "
        + "analysis_raster_id: "
        + analysis_raster_id
        + ", "
        + "contextual_raster_ids: "
        + str(contextual_raster_ids)
        + ", "
        + "aggregate_raster_ids: "
        + str(aggregate_raster_ids)
        + ", "
        + "filter_raster_id: "
        + str(filter_raster_id)
        + ", "
        + "filter_intervals: "
        + str(filter_intervals)
        + ", "
        + "geom: "
        + str(json.dumps(mapping(geom)))
    )

    result = dict()

    unique_raster_sources = get_raster_id_array(
        analysis_raster_id,
        contextual_raster_ids,
        aggregate_raster_ids,
        filter_raster_id,
        unique=True,
    )

    raster_windows = read_windows_parallel(
        unique_raster_sources, geom, analysis_raster_id
    )

    # fail if analysis layer is empty
    if raster_windows[analysis_raster_id].data.size == 0:
        raise RasterReadException(
            "Analysis raster `" + analysis_raster_id + "` returned empty array"
        )

    mean_area = get_area(geom.centroid.y) / 10000

    # other layers we'll just use raster with all values == 0
    for raster_id in get_raster_id_array(
        contextual_raster_ids, aggregate_raster_ids, filter_raster_id
    ):
        if raster_windows[raster_id].data.size == 0:
            raster_windows[raster_id] = RasterWindow(
                np.zeros(raster_windows[analysis_raster_id].data.shape, dtype=np.uint8),
                raster_windows[analysis_raster_id].shifted_affine,
                raster_windows[raster_id].no_data,
            )

    analysis_data, shifted_affine, no_data = raster_windows[analysis_raster_id]

    if filter_raster_id:
        filter_mask = _get_filter(
            raster_windows[filter_raster_id].data, filter_intervals
        )
        raster_windows["filter"] = RasterWindow(filter_mask, None, None)

    # start by masking geometry onto analysis layer (where mask=True indicates the geom intersects)
    mask = mask_geom_on_raster(analysis_data, shifted_affine, geom)

    logger.debug("Successfully converted extracted data to dataframe")

    # apply initial analysis, grouping by all but aggregate fields (these may be later further grouped)
    analysis_groupby_fields = get_raster_id_array(
        analysis_raster_id, contextual_raster_ids, "filter"
    )
    analysis_result = _analysis(
        analyses,
        raster_windows,
        analysis_groupby_fields,
        aggregate_raster_ids,
        mean_area,
        mask,
    )
    # select=[loss, wdpa, ifl, sum(biomass) as total_biomass, (tcd_2000 > 30) as tree_cover]
    detailed_table = _get_detailed_table(analysis_result, no_data, analysis_raster_id)
    summary_table = _get_summary_table(
        analyses,
        analysis_result,
        no_data,
        analysis_raster_id,
        contextual_raster_ids,
        aggregate_raster_ids,
    )

    result["detailed_table"] = detailed_table.to_dict()
    result["summary_table"] = summary_table.to_dict()

    logger.debug("Successfully ran analysis=" + str(analyses))
    logger.info("Ran analysis with result: " + json.dumps(result))

    return result


def _get_detailed_table(analysis_result, no_data_value, analysis_raster_id):
    no_data_filter = analysis_result[analysis_raster_id] != no_data_value
    passes_filter = analysis_result[FILTER_FIELD]

    # filter out rows that don't pass filter or where analysis layer is NoData, and remove the
    # "filter" field from final result
    return (
        analysis_result[passes_filter & no_data_filter]
        .drop(columns=[FILTER_FIELD])
        .reset_index(drop=True)
    )


def _get_summary_table(
    analyses,
    analysis_result,
    no_data_value,
    analysis_raster_id,
    contextual_raster_ids,
    aggregate_raster_ids,
):
    summary_table = analysis_result.copy()

    if "area" in analyses:
        # create new column that just copies area column, but sets any row with filter=false to 0
        summary_table[FILTERED_AREA_FIELD] = (
            summary_table[AREA_FIELD] * summary_table[FILTER_FIELD]
        )

        # create new column that copies filtered column, but sets any row where analysis layer is NoData to 0
        layer_area_field = LAYER_AREA_FIELD.format(raster_id=analysis_raster_id)
        summary_table[layer_area_field] = summary_table[FILTERED_AREA_FIELD] * (
            summary_table[analysis_raster_id] != no_data_value
        )

        drop_columns = [analysis_raster_id, FILTER_FIELD] + aggregate_raster_ids
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
            return (
                pd.DataFrame(summary_table.sum()).transpose().drop(columns=drop_columns)
            )
    else:
        return None


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
    return np.bincount(linear_indices, weights=np.extract(mask, data), minlength=bins)


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


def _mask_by_threshold(raster, threshold):
    return raster > threshold


def _mask_by_nodata(raster, no_data):
    return raster != no_data


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
