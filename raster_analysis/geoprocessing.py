import logging

import numpy as np
import json

from raster_analysis.geodesy import get_area
from raster_analysis.io import mask_geom_on_raster, read_windows_parallel, RasterWindow

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
CO2_FACTOR = 0.5 * 44 / 12  # used to calculate emissions from biomass layer


@xray_recorder.capture("Geoprocessing Analysis")
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

    if not raster_windows:
        return _get_empty_result(
            analyses, analysis_raster_id, contextual_raster_ids, aggregate_raster_ids
        )

    mask = _get_mask(
        geom, raster_windows, analysis_raster_id, extent_layer_id, start, end
    )

    # apply initial analysis, grouping by all but aggregate fields (these may be later further grouped)
    analysis_groupby_fields = get_raster_id_array(
        analysis_raster_id, contextual_raster_ids
    )

    result = _analysis(
        analyses,
        raster_windows,
        analysis_groupby_fields,
        aggregate_raster_ids,
        mean_area,
        mask,
    )

    for col, arr in result.items():
        result[col] = arr.tolist()

    logger.debug(f"Successfully ran analysis={analyses}")
    logger.info(f"Ran analysis with result: {result}")

    return result


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

    # TODO emissions should exist as a virtual dataset in our data lake, with metadata on how to calculate it
    # TODO for now, just hardcoded here
    if "emissions" in unique_raster_sources:
        unique_raster_sources.remove("emissions")

        if "biomass" not in unique_raster_sources:
            unique_raster_sources.append("biomass")

    raster_windows = read_windows_parallel(unique_raster_sources, geom)

    # if nothing could be read, nothing to return
    if all([window.data.size == 0 for window in raster_windows.values()]):
        return None

    sample_layer, shifted_affine, no_data = list(raster_windows.values())[0]

    # other layers we'll just use raster with all values == 0
    for raster_id in get_raster_id_array(
        contextual_raster_ids, aggregate_raster_ids, extent_layer_id
    ):
        if raster_windows[raster_id].data.size == 0:
            raster_windows[raster_id] = RasterWindow(
                np.zeros(raster_windows[analysis_raster_id].data.shape, dtype=np.uint8),
                shifted_affine,
                0,
            )

    if extent_layer_id:
        raster_windows[extent_layer_id] = RasterWindow(
            _mask_by_threshold(raster_windows[extent_layer_id].data, threshold),
            raster_windows[extent_layer_id].shifted_affine,
            raster_windows[extent_layer_id].no_data,
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
    """
    The analysis algorithm ended up using some obscure NumPy functions to optimize speed and memory usage,
    so an explanation of what's going on here:

    ravel flattens gives a view of the array as flattened rather than actually allocating a new array
    and flattening it, since our 2D arrays are huge

    ravel_multi_index will generate "linear indices" of all the layers we want to group by. A linear index is just
    an array index, but for multi-dimensional arrays, a single number saying how many indices you have to walk past
    to get to that index. E.g. if we have a 2D array, with each dimension length 4, the index [2][1] would have a
    linear index of 10, because you have to walk two full rows (8) + two to get to column [1].

    So what's going on here is we're pretending like each layer is a dimension in an n-dimensional array,
    and each value is actually an array index in that dimension (where the dimension length is the max
    possible value of that layer). By getting the linear index for each layer value combination at each
    pixel, we're getting a unique integer for that combination.

    We need this because np.unique runs super slowly on vectors (i.e. if we just stacked all the layers
    and called np.unique) and runs wayyy faster on just a 1D array of integers.

    unravel_index allows us to unpack the unique combinations back into layer values once we run np.unique.

    For aggregate columns, we're using np.bincount on the linear indices, which will sum the number of times
    each unique value occurs. By adding the aggregate layer as weights, instead of just adding 1 for each,
    it'll add the value of the aggregate layer where the array indices match.

    Because this is counting bins, and the linear indices can be a huge range values, we actually only want to bins
    for values that exist in the array. To do that, we call np.digitize, which provides an incrementing integer
    value from 0 for each unique value. So [5, 4, 8, 4] would become [0, 1, 2, 1]. By doing this, we still will
    correctly aggregate for each unique value, and the result will line up perfectly with our result from np.unique.
    (np.unique actually has a param called return_inverse that will return this, but it's slower so only getting this
    array if we actually have aggregate layers).

    For reference on future design decisions, this was originally use pandas, but pandas allocating lots of
    temporary arrays that exploded the memory usage (e.g. with 2 GB of layer data, it would allocate an
    additional 2.5 GB during processing). I made the choice here to very sparingly allocate new arrays
    and re-use existing ones as much as possible to keep us under lambda limits and increase speed.
    """
    reporting_columns = [
        np.ravel(raster_windows[raster_id].data) for raster_id in reporting_raster_ids
    ]
    column_maxes = [col.max() + 1 for col in reporting_columns]

    linear_indices = _get_linear_indices(reporting_columns, column_maxes, mask)

    unique_values, counts = _unique(linear_indices)
    unique_value_combinations = np.unravel_index(unique_values, column_maxes)

    result = dict(zip(reporting_raster_ids, unique_value_combinations))

    if "count" in analyses:
        result[COUNT_FIELD] = counts

    if "area" in analyses:
        result[AREA_FIELD] = counts * mean_area

    if aggregate_raster_ids:
        linear_indices = get_inverse(linear_indices, unique_values)

        for raster_id in aggregate_raster_ids:
            if raster_id == "emissions":
                result["emissions"] = (
                    _get_sum(
                        linear_indices,
                        mask,
                        raster_windows["biomass"].data,
                        counts.size,
                    )
                    * mean_area
                    * CO2_FACTOR
                )
            else:
                result[raster_id] = _get_sum(
                    linear_indices, mask, raster_windows[raster_id].data, counts.size
                )

                if raster_id == "biomass":
                    result[raster_id] *= mean_area

    return result


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


def _get_empty_result(
    analyses, analysis_raster_id, contextual_raster_ids, aggregate_raster_ids
):
    unique_raster_sources = get_raster_id_array(
        analysis_raster_id, contextual_raster_ids, aggregate_raster_ids, unique=True
    )

    empty_results = [[] for i in range(0, len(unique_raster_sources))]
    result = dict(zip(unique_raster_sources, empty_results))

    for analysis in analyses:
        result[analysis] = []

    return result


def _result_to_json(result):
    for name, arr in enumerate(result):
        result[name] = arr.tolist()
