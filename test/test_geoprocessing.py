import numpy as np
import pandas as pd
from unittest import mock
from shapely.geometry import Polygon

from raster_analysis.geoprocessing import analysis
from raster_analysis.io import RasterWindow

GEOM = Polygon([(0, 0), (1, 1), (1, 0)])


@mock.patch("raster_analysis.geoprocessing.read_windows_parallel")
@mock.patch("raster_analysis.geoprocessing.mask_geom_on_raster")
def test_only_analysis(mock_geom_on_raster, mock_read_windows):
    mock_read_windows.return_value = {
        "ras0": RasterWindow(
            np.array(
                [
                    [0, 1, 2, 1, 0],
                    [0, 1, 2, 1, 0],
                    [0, 1, 2, 1, 0],
                    [0, 1, 2, 1, 0],
                    [0, 1, 2, 1, 0],
                ]
            ),
            None,
            0,
        )
    }

    mock_geom_on_raster.return_value = np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
        ]
    )

    results = analysis(GEOM, ["count"], "ras0")

    assert len(results["count"]) == 2
    assert results["count"][0] == 6
    assert results["count"][1] == 3


@mock.patch("raster_analysis.geoprocessing.read_windows_parallel")
@mock.patch("raster_analysis.geoprocessing.mask_geom_on_raster")
def test_analysis_with_extent_filter_and_sum(mock_geom_on_raster, mock_read_windows):
    mock_read_windows.return_value = {
        "ras0": RasterWindow(
            np.array(
                [
                    [0, 1, 2, 1, 0],
                    [0, 1, 2, 1, 0],
                    [0, 1, 2, 1, 0],
                    [0, 1, 2, 1, 0],
                    [0, 1, 2, 1, 0],
                ]
            ),
            None,
            0,
        ),
        "ras_sum": RasterWindow(
            np.array(
                [
                    [0, 1.2, 2.2, 3.2, 2.2],
                    [0, 1.2, 2.2, 3.2, 2.2],
                    [0, 1.2, 2.2, 3.2, 2.2],
                    [0, 1.2, 2.2, 3.2, 2.2],
                    [0, 1.2, 2.2, 3.2, 2.2],
                ]
            ),
            None,
            0,
        ),
        "tcd_2000": RasterWindow(
            np.array(
                [
                    [50, 31, 29, 30, 10],
                    [10, 40, 40, 10, 10],
                    [10, 10, 40, 40, 40],
                    [40, 40, 10, 40, 40],
                    [40, 40, 10, 10, 40],
                ]
            ),
            None,
            0,
        ),
    }

    mock_geom_on_raster.return_value = np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
        ]
    )

    results = analysis(
        GEOM,
        ["count"],
        "ras0",
        extent_year=2000,
        threshold=30,
        aggregate_raster_ids=["ras_sum"],
    )

    assert len(results["count"]) == 2
    assert results["count"][0] == 4
    assert results["ras_sum"][0] == 8.8
    assert results["count"][1] == 2
    assert results["ras_sum"][1] == 4.4


# TODO: what to do with context layer NoData values?
@mock.patch("raster_analysis.geoprocessing.read_windows_parallel")
@mock.patch("raster_analysis.geoprocessing.mask_geom_on_raster")
@mock.patch("raster_analysis.geoprocessing.get_area")
def test_analysis_with_context_and_time_filters_and_area(
    mock_get_area, mock_geom_on_raster, mock_read_windows
):
    mock_read_windows.return_value = {
        "ras0": RasterWindow(
            np.array(
                [
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                ]
            ),
            None,
            0,
        ),
        "ras1": RasterWindow(
            np.array(
                [
                    [3, 1, 2, 1, 2],
                    [3, 1, 2, 3, 2],
                    [3, 1, 2, 1, 1],
                    [3, 1, 2, 3, 1],
                    [3, 1, 2, 1, 1],
                ]
            ),
            None,
            3,
        ),
        "ras2": RasterWindow(
            np.array(
                [
                    [100, 200, 300, 400, 500],
                    [100, 200, 300, 400, 500],
                    [100, 200, 300, 400, 500],
                    [100, 200, 300, 400, 500],
                    [100, 200, 300, 400, 500],
                ]
            ),
            None,
            0,
        ),
    }

    geom_on_raster = np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
        ]
    )

    mock_geom_on_raster.return_value = np.copy(geom_on_raster)
    mock_get_area.return_value = 20000

    results = analysis(
        GEOM,
        ["count", "area"],
        "ras0",
        contextual_raster_ids=["ras1", "ras2"],
        start=2,
        end=4,
    )
    df = pd.DataFrame.from_dict(results)

    assert len(results["count"]) == 4
    assert np.all(df.loc[0] == [2, 1, 200, 2, 4])
    assert np.all(df.loc[1] == [3, 2, 300, 3, 6])
    assert np.all(df.loc[2] == [4, 1, 400, 2, 4])
    assert np.all(df.loc[3] == [4, 3, 400, 2, 4])

    # reset mock mask
    mock_geom_on_raster.return_value = np.copy(geom_on_raster)

    results = analysis(
        GEOM, ["count", "area"], "ras0", contextual_raster_ids=["ras1", "ras2"], start=4
    )
    df = pd.DataFrame.from_dict(results)

    assert len(results["count"]) == 4
    assert np.all(df.loc[0] == [4, 1, 400, 2, 4])
    assert np.all(df.loc[1] == [4, 3, 400, 2, 4])
    assert np.all(df.loc[2] == [5, 1, 500, 3, 6])
    assert np.all(df.loc[3] == [5, 2, 500, 2, 4])

    # reset mock mask
    mock_geom_on_raster.return_value = np.copy(geom_on_raster)

    results = analysis(
        GEOM, ["count", "area"], "ras0", contextual_raster_ids=["ras1", "ras2"], end=2
    )
    df = pd.DataFrame.from_dict(results)

    assert len(results["count"]) == 2
    assert np.all(df.loc[0] == [1, 3, 100, 1, 2])
    assert np.all(df.loc[1] == [2, 1, 200, 2, 4])


# TODO: best way to get extent summary?
@mock.patch("raster_analysis.geoprocessing.read_windows_parallel")
@mock.patch("raster_analysis.geoprocessing.mask_geom_on_raster")
@mock.patch("raster_analysis.geoprocessing.get_area")
def test_summary(mock_get_area, mock_geom_on_raster, mock_read_windows):
    mock_read_windows.return_value = {
        "ras1": RasterWindow(
            np.array(
                [
                    [3, 1, 2, 1, 2],
                    [3, 1, 2, 3, 2],
                    [3, 1, 2, 1, 1],
                    [3, 1, 2, 3, 1],
                    [3, 1, 2, 1, 1],
                ]
            ),
            None,
            3,
        ),
        "ras2": RasterWindow(
            np.array(
                [
                    [100, 200, 300, 400, 500],
                    [100, 200, 300, 400, 500],
                    [100, 200, 300, 400, 500],
                    [100, 200, 300, 400, 500],
                    [100, 200, 300, 400, 500],
                ]
            ),
            None,
            0,
        ),
    }

    geom_on_raster = np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
        ]
    )

    mock_geom_on_raster.return_value = np.copy(geom_on_raster)
    mock_get_area.return_value = 20000

    results = analysis(GEOM, ["area"], contextual_raster_ids=["ras1", "ras2"])
    df = pd.DataFrame.from_dict(results)

    assert len(results["area"]) == 7
    assert np.all(df.loc[0] == [1, 200, 4])
    assert np.all(df.loc[1] == [1, 400, 4])
    assert np.all(df.loc[2] == [1, 500, 6])
    assert np.all(df.loc[3] == [2, 300, 6])
    assert np.all(df.loc[4] == [2, 500, 4])
    assert np.all(df.loc[5] == [3, 100, 2])
    assert np.all(df.loc[6] == [3, 400, 4])
