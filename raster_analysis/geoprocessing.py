import logging
from collections import namedtuple

import numpy as np
import pandas as pd
import json
import gc

from raster_analysis.geodesy import get_area
from raster_analysis.io import mask_geom_on_raster, read_windows_parallel, RasterWindow
from raster_analysis.exceptions import RasterReadException

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

    get_area_raster = "area" in analyses or density_raster_ids

    raster_windows = read_windows_parallel(
        unique_raster_sources, geom, analysis_raster_id, get_area_raster=get_area_raster
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

    mean_area = get_area((geom.bounds[3] - geom.bounds[1]) / 2) / 10000
    analysis_data, shifted_affine, no_data = raster_windows[analysis_raster_id]

    # start by masking geometry onto analysis layer (where mask=True indicates the geom intersects)
    geom_mask = mask_geom_on_raster(analysis_data, shifted_affine, geom)

    # save geom_mask * filter_mask calculation, since we may need it again for the area summary
    filtered_geom_mask = None

    if filter_raster_id:
        filter_mask = _get_filter(
            raster_windows[filter_raster_id].data, filter_intervals
        )
        filtered_geom_mask = geom_mask * filter_mask
        mask = filtered_geom_mask * _mask_by_nodata(analysis_data, no_data)
    else:
        mask = geom_mask * _mask_by_nodata(analysis_data, no_data)

    logger.debug("Successfully created full mask.")

    if get_area_summary:
        # get first column of area matrix to use as area vector
        pixel_areas = raster_windows[AREA_FIELD].data[:, 0]

        result["area_summary__ha"] = _get_area_summary(
            geom_mask,
            contextual_raster_ids,
            raster_windows,
            pixel_areas,
            filtered_geom_mask,
        )

    # extract the analysis raster data first since it'll be used to get geometry mask
    extracted_data = {analysis_raster_id: _extract(mask, analysis_data)}

    # extract data from aggregate and contextual layers, applying geometry mask and appending to dict
    for raster_id in contextual_raster_ids + aggregate_raster_ids:
        extracted_data[raster_id] = _extract(mask, raster_windows[raster_id].data)

    if get_area_raster:
        extracted_data[AREA_FIELD] = _extract(mask, raster_windows[AREA_FIELD].data)

    # unbind/garbage collect all NumPy arrays we're done with to free up space
    del analysis_data
    del raster_windows
    del mask

    if filter_raster_id:
        del filter_mask
        del filtered_geom_mask

    gc.collect()

    # convert to pandas DataFrame for analysis
    extracted_df = pd.DataFrame(extracted_data)
    logger.debug("Successfully converted extracted data to dataframe")

    # apply initial analysis, grouping by all but aggregate fields (these may be later further grouped)
    analysis_groupby_fields = get_raster_id_array(
        analysis_raster_id, contextual_raster_ids
    )
    analysis_result = _analysis(
        analyses, extracted_df, analysis_groupby_fields, aggregate_raster_ids, mean_area
    )
    logger.debug("Successfully ran analysis=" + str(analyses))

    result["results"] = analysis_result.to_dict()

    logger.info("Ran analysis with result: " + json.dumps(result))

    return result


@xray_recorder.capture("Pandas Analysis")
def _analysis(analyses, df, reporting_raster_ids, aggregate_raster_ids, mean_area):
    result = None

    # area is just count * mean area, so always get count if analysis has count or area
    if "count" in analyses:
        # to get both count and sum, we have to do some wonky stuff with pandas to get both
        # aggregations at the same time.

        if "area" in analyses or "sum" in analyses:
            if "sum" in analyses:
                if len(aggregate_raster_ids) >= 0:
                    raise Exception("No aggregate rasters specified for sum analysis")

                agg_fields = aggregate_raster_ids

                if "area" in analyses:
                    agg_fields += [AREA_FIELD]
            else:
                agg_fields = [AREA_FIELD]

            agg = {agg_fields[0]: ["count", "sum"]}

            # then explicitly ask for sum of all other agg layers
            if len(agg_fields) > 0:
                for agg_field_id in agg_fields[1:]:
                    agg[agg_field_id] = "sum"

            result = df.groupby(reporting_raster_ids).agg(agg).reset_index()

            # now rename columns so that the count agg field is just called 'count'
            result.columns = reporting_raster_ids + [COUNT_FIELD] + agg_fields
            logger.debug("Successfully calculated count and sum")
        else:
            # otherwise use pandas built-in count agg function
            result = (
                df.groupby(reporting_raster_ids).size().reset_index(name=COUNT_FIELD)
            )
            logger.debug("Successfully calculated count")
    # if only sum or area needed, just use pandas built-in sum function
    else:
        result = df.groupby(reporting_raster_ids).sum().reset_index()
        logger.debug("Successfully calculated " + str(analyses))

    return result


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
