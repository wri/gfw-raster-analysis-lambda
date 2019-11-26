from raster_analysis.geoprocessing import analysis
from shapely.geometry import shape
import test.test_geoms as test_geoms
import pandas as pd
import pytest


def test_basic_verification():
    result = analysis(
        geom=shape(test_geoms.BRAZIL_SMALL["geometry"]),
        analysis_raster_id="loss",
        contextual_raster_ids=["wdpa", "ifl"],
        aggregate_raster_ids=["biomass"],
        filter_raster_id="tcd_2000",
        filter_intervals=[[30, 100]],
        analyses=["count", "area"],
    )

    detailed = pd.DataFrame(result["detailed_table"])
    summary = pd.DataFrame(result["summary_table"])

    assert (
        pytest.approx(summary["area"].sum(), 0.01)
        == test_geoms.BRAZIL_SMALL["total_area"]
    )
    assert (
        pytest.approx(summary["filtered_area"].sum(), 0.01)
        == test_geoms.BRAZIL_SMALL["tcd_2000_area"]
    )
    assert (
        pytest.approx(detailed["area"].sum(), 0.01)
        == test_geoms.BRAZIL_SMALL["loss_area"]
    )
    assert (
        pytest.approx(summary["loss_area"].sum(), 0.01)
        == test_geoms.BRAZIL_SMALL["loss_area"]
    )


def test_pantropic():
    return


def test_non_pantropic():
    return


def test_across_tile_boundaries():
    return


def test_middle_of_nowhere():
    return


def test_uncommon_shapes():
    return


def test_float_sum():
    return


def test_weird_thing():
    return
