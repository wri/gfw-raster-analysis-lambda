from raster_analysis.grid import get_raster_url
from raster_analysis.io import (
    read_window,
    read_window_ignore_missing,
    mask_geom_on_raster,
)
from raster_analysis.geodesy import get_area
from collections import namedtuple
import pandas as pd
import logging
import numpy as np

Filter = namedtuple("Filter", "raster_id threshold")


def analysis(
    geom,
    analysis_raster_id,
    contextual_raster_ids=[],
    aggregate_raster_ids=[],
    filters=[],
    analysis="count",
):
    """
    Supported analysis:
        area: calculates geodesic area for unique pixel combinations across all input rasters
        sum: Sums values for all floating point rasters for unique pixel combintions of non-floating point rasters
        count: Counts occurrences of unique pixel combinations across all input rasters

    If threshold is > 0, the 2nd and 3rd raster need to be tcd2000 and tcd2010.
        It will use tcd2000 layer as additional mask and add
    """

    logging.info(
        "[INFO][RasterAnalysis] Running analysis:  `"
        + analysis
        + "` across layer `"
        + analysis_raster_id
        + "` with geometry: "
        + geom.to_wkt()
    )

    result = dict()

    mean_area = get_area((geom.bounds[3] - geom.bounds[1]) / 2) / 10000

    raster = get_raster_url(analysis_raster_id)
    data, geom_mask, _, no_data = mask_geom_on_raster(geom, raster)

    if geom_mask.any():
        mask = _generate_full_mask(data, geom_mask, no_data, geom, filters)

        # extract the analysis raster data first since it'll be used as a reference
        extracted_data = {analysis_raster_id: np.extract(mask, data)}

        # extract data from contextual layers and aggregate layers and append to dict
        extracted_data.update(
            _extract_raster_data(
                contextual_raster_ids + aggregate_raster_ids,
                geom,
                mask,
                len(extracted_data[analysis_raster_id]),
            )
        )

        extracted_df = pd.DataFrame(extracted_data)
        reporting_raster_ids = [analysis_raster_id] + contextual_raster_ids

        analysis_result = _analysis(
            analysis, extracted_df, reporting_raster_ids, mean_area
        )

        result["data"] = analysis_result.to_dict()
        result["extent"] = mask.sum() * mean_area

        return result
    else:
        logging.debug(
            "[DEBUG][RasterAnalysis] Skipping analysis because entire geometry is masked"
        )
        return dict()


def _analysis(analysis, df, reporting_raster_ids, mean_area):
    if analysis == "count":
        return _count(df, reporting_raster_ids)
    elif analysis == "area":
        return _area(df, reporting_raster_ids, mean_area)
    elif analysis == "sum":
        return _sum(df, reporting_raster_ids)
    else:
        raise ValueError("Unknown analysis: " + analysis)


def _count(df, raster_ids):
    return df.groupby(raster_ids).size().reset_index(name="count")


def _area(df, raster_ids, mean_area):
    result = df.groupby(raster_ids).size().reset_index(name="area")
    result.area = result.area.multiply(mean_area)
    return result


def _sum(df, raster_ids):
    return df.groupby(raster_ids).sum().reset_index()


def _generate_full_mask(data, geom_mask, no_data, geom, filters):
    value_mask = _mask_by_nodata(data, no_data)
    mask = geom_mask * value_mask
    return _apply_filters(filters, mask, geom)


def _extract_raster_data(raster_ids, geom, mask, missing_length):
    extracted_data = dict()

    for raster_id in raster_ids:
        raster = get_raster_url(raster_id)
        data, _, _ = read_window_ignore_missing(raster, geom)
        if data.any():
            extracted_data[raster_id] = np.extract(mask, data)
        else:
            extracted_data[raster_id] = np.zeros(len(missing_length))

    return extracted_data


def _mask_by_threshold(raster, threshold):
    return raster > threshold


def _mask_by_nodata(raster, no_data):
    return raster != no_data


def _apply_filters(filters, mask, geom):
    if filters:
        for curr_filter in filters:
            curr_filter_url = get_raster_url(curr_filter.raster_id)
            curr_filter_mask = _mask_by_threshold(
                read_window(curr_filter_url, geom)[0], curr_filter.threshold
            )

            mask *= curr_filter_mask

    return mask
