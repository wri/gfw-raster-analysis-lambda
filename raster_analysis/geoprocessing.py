import logging
from collections import namedtuple

import numpy as np
import pandas as pd
import json
import gc

from raster_analysis.geodesy import get_area
from raster_analysis.grid import get_raster_url
from raster_analysis.io import (
    mask_geom_on_raster,
    read_window_ignore_missing,
    read_windows_parallel,
)

from aws_xray_sdk.core import xray_recorder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AREA_FIELD = "area"
COUNT_FIELD = "count"
SUM_FIELD = "sum"
FILTERED_AREA_FIELD = "filtered_area"
LAYER_AREA_FIELD = "{raster_id}_area"
FILTER_FIELD = "filter"

Filter = namedtuple("Filter", "raster_id threshold")


def analysis(
    geom,
    analysis_raster_id,
    contextual_raster_ids=[],
    aggregate_raster_ids=[],
    filters=[],
    analyses=["count", "area"],
    get_summary_table=False,
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
        + "filters: "
        + str(filters)
        + ", "
        + "geom: "
        + str(geom.to_wkt())
    )

    result = dict()

    filter_raster_ids = [filt.raster_id for filt in filters]
    raster_windows = read_windows_parallel(
        [analysis_raster_id]
        + contextual_raster_ids
        + aggregate_raster_ids
        + filter_raster_ids,
        geom,
    )

    mean_area = get_area((geom.bounds[3] - geom.bounds[1]) / 2) / 10000
    analysis_data, shifted_affine, no_data = raster_windows[analysis_raster_id]

    # start by masking geometry onto analysis layer (where mask=True indicates the geom intersects)
    geom_mask = mask_geom_on_raster(analysis_data, shifted_affine, geom)
    total_filter = _get_total_filter(raster_windows, filters, analysis_data.shape)
    mask = geom_mask * total_filter * _mask_by_nodata(analysis_data, no_data)

    logger.debug("Successfully masked geometry onto analysis layer.")

    # extract the analysis raster data first since it'll be used to get geometry mask
    extracted_data = {analysis_raster_id: _extract(mask, analysis_data)}

    # extract data from aggregate and contextual layers, applying geometry mask and appending to dict
    for raster_id in contextual_raster_ids + aggregate_raster_ids:
        extracted_data[raster_id] = _extract(mask, raster_windows[raster_id].data)

    del analysis_data
    del raster_windows
    del total_filter

    gc.collect()

    # convert to pandas DataFrame for analysis
    extracted_df = pd.DataFrame(extracted_data)
    logger.debug("Successfully converted extracted data to dataframe")

    # apply initial analysis, grouping by all but aggregate fields (these may be later further grouped)
    analysis_groupby_fields = [analysis_raster_id] + contextual_raster_ids
    analysis_result = _analysis(
        analyses, extracted_df, analysis_groupby_fields, aggregate_raster_ids, mean_area
    )
    logger.debug("Successfully ran analysis=" + str(analyses))

    if get_summary_table:
        # TODO Figure out performant way to generate summary table
        raise Exception("Not Implemented")
    else:
        result = analysis_result.to_dict()

    logger.info("Ran analysis with result: " + json.dumps(result))

    return result


@xray_recorder.capture("Pandas Analysis")
def _analysis(analyses, df, reporting_raster_ids, aggregate_raster_ids, mean_area):
    result = None

    # area is just count * mean area, so always get count if analysis has count or area
    if "count" or "area" in analyses:
        # to get both count and sum, we have to do some wonky stuff with pandas to get both
        # aggregations at the same time.
        if "sum" in analyses and len(aggregate_raster_ids) > 0:
            # During sum of the first agg layer, also get get the count
            agg = {aggregate_raster_ids[0]: ["count", "sum"]}

            # then explicitly ask for sum of all other agg layers
            if len(aggregate_raster_ids) > 0:
                for raster_id in aggregate_raster_ids[1:]:
                    agg[raster_id] = "sum"

            result = df.groupby(reporting_raster_ids).agg(agg).reset_index()

            # now rename columns so that the count agg field is just called 'count'
            result.columns = reporting_raster_ids + [COUNT_FIELD] + aggregate_raster_ids
            logger.debug("Successfully calculated count and sum")
        else:
            # otherwise use pandas built-in count agg function
            result = (
                df.groupby(reporting_raster_ids).size().reset_index(name=COUNT_FIELD)
            )
            logger.debug("Successfully calculated count")

    if "area" in analyses:
        # use previously calculated count column to generate area column
        result[AREA_FIELD] = result[COUNT_FIELD].multiply(mean_area)

        # if count was just a temp column to calculate area, drop it
        if "count" not in analyses:
            result = result.drop(columns=[COUNT_FIELD])

        logger.debug("Successfully calculated area")

    # if we never needed to calculate sum and count at same time, just use for pandas built-in sum agg function
    if "sum" in analyses and "count" not in analyses and "area" not in analyses:
        result = df.groupby(reporting_raster_ids).sum().reset_index()
        logger.debug("Successfully calculated sum")

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
    analyses, analysis_result, no_data_value, analysis_raster_id, contextual_raster_ids
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

        # aggregate by combos of contextual layer values, and drop fields we don't want in the final result
        return (
            summary_table.groupby(contextual_raster_ids)
            .sum()
            .reset_index()
            .drop(columns=[analysis_raster_id, FILTER_FIELD])
        )
    else:
        return None


@xray_recorder.capture("Read and Extract Raster Data")
def _extract_raster_data(raster_ids, geom, mask, missing_length):
    extracted_data = dict()

    for raster_id in raster_ids:
        raster = get_raster_url(raster_id)
        data, _, _ = read_window_ignore_missing(raster, geom)
        if data.any():
            extracted_data[raster_id] = _extract(mask, data)
            logger.debug("Successfully masked geometry onto layer=" + raster_id)
        else:
            extracted_data[raster_id] = np.zeros(missing_length)

    return extracted_data


def _mask_by_threshold(raster, threshold):
    return raster > threshold


def _mask_by_nodata(raster, no_data):
    return raster != no_data


def _get_total_filter(raster_windows, filters, shape):
    total_filter = np.ones(shape, dtype=np.bool)

    if filters:
        for curr_filter in filters:
            curr_filter_mask = _mask_by_threshold(
                raster_windows[curr_filter.raster_id].data, curr_filter.threshold
            )
            logger.debug(
                "Successfully masked threshold="
                + str(curr_filter.threshold)
                + " onto filter layer="
                + curr_filter.raster_id
            )

            total_filter = curr_filter_mask * total_filter

    logger.debug("Successfully created aggregate filter")
    return total_filter


@xray_recorder.capture("NumPy Extract")
def _extract(mask, data):
    return np.extract(mask, data)
