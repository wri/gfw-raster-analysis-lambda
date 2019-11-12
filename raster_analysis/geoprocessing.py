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

from aws_xray_sdk.core import xray_recorder

import resource

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AREA_FIELD = "area"
COUNT_FIELD = "count"
SUM_FIELD = "sum"


def analysis(
    geom,
    analysis_raster_id,
    contextual_raster_ids=[],
    aggregate_raster_ids=[],
    filter_raster_id=None,
    filter_intervals=None,
    density_raster_ids=[],
    analyses=["count", "area"],
    get_area_summary=False,
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
        + str(geom.to_wkt())
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

    # use area raster to turn raster with density values into a full values
    for raster_id in density_raster_ids:
        undensified_data = (
            raster_windows[raster_id].data * raster_windows[AREA_FIELD].data
        )
        raster_windows[raster_id] = RasterWindow(
            undensified_data,
            raster_windows[raster_id].shifted_affine,
            raster_windows[raster_id].no_data,
        )
        gc.collect()  # immediately collect dereferenced density array

    mean_area = get_area(geom.centroid.y) / 10000
    analysis_data, shifted_affine, no_data = raster_windows[analysis_raster_id]

    # start by masking geometry onto analysis layer (where mask=True indicates the geom intersects)
    geom_mask = mask_geom_on_raster(analysis_data, shifted_affine, geom)

    if filter_raster_id:
        filter_mask = _get_filter(
            raster_windows[filter_raster_id].data, filter_intervals
        )

    mask = geom_mask
    raster_windows["filter"] = RasterWindow(filter_mask, None, None)

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

    logger.debug("Successfully ran analysis=" + str(analyses))
    logger.info("Ran analysis with result: " + json.dumps(result))

    return analysis_result.to_dict()


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
    linear_indices = np.compress(
        np.ravel(mask),
        np.ravel_multi_index(reporting_columns, column_maxes).astype(np.uint32),
    )

    unique_values, counts = np.unique(linear_indices, return_counts=True)
    unique_value_combinations = np.unravel_index(unique_values, column_maxes)

    result_dict = dict(zip(reporting_raster_ids, unique_value_combinations))

    if "count" in analyses:
        result_dict[COUNT_FIELD] = counts

    if "area" in analyses:
        result_dict[AREA_FIELD] = counts * mean_area

    if aggregate_raster_ids:
        linear_indices = get_inverse(linear_indices, unique_values)

        for raster_id in aggregate_raster_ids:
            result_dict[raster_id] = np.bincount(
                linear_indices,
                weights=np.extract(mask, raster_windows[raster_id].data),
                minlength=counts.size,
            )

    return pd.DataFrame(result_dict)


@xray_recorder.capture("Get Inverse")
def get_inverse(linear_indices, unique_values):
    return np.digitize(linear_indices, unique_values, right=True)


@xray_recorder.capture("Get Area Summary")
def _get_area_summary(
    geom_mask,
    contextual_layer_ids,
    raster_windows,
    area_vector,
    filtered_geom_mask=None,
):
    area_summary = dict()
    area_summary["total"] = (geom_mask.sum(axis=1) * area_vector).sum()

    if filtered_geom_mask.size != 0:
        area_summary["filtered"] = (filtered_geom_mask.sum(axis=1) * area_vector).sum()

    mask = filtered_geom_mask if filtered_geom_mask.size != 0 else geom_mask
    for layer_id in contextual_layer_ids:
        area_summary[layer_id] = (
            (raster_windows[layer_id].data * mask).sum(axis=1) * area_vector
        ).sum()

    return area_summary


@xray_recorder.capture("Create Filter")
def _get_filter(filter_raster, filter_intervals):
    filter = np.zeros(filter_raster.shape, dtype=np.bool)

    for interval in filter_intervals:
        filter += (filter_raster >= interval[0]) * (filter_raster < interval[1])

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
