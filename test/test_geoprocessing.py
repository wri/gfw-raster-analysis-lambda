from unittest import mock

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from raster_analysis import geoprocessing
from raster_analysis.geoprocessing import Filter

A = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
B = np.array([[2, 3, 4], [3, 4, 5], [4, 5, 6]])
C = np.array([[3, 4, 5], [4, 5, 6], [5, 6, 7]])
F1 = np.array([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.5]])
F2 = np.array([[3.0, 4.0, 5.0], [4.0, 5.0, 6.0], [5.0, 6.0, 7.5]])
MASK = np.array([[False, False, True], [False, True, True], [True, True, True]])
FILTER = np.array([[50, 10, 35], [40, 80, 30], [10, 60, 90]])

AREA = 769.288482
THRESHOLD = 30

GEOMETRY = Polygon([(0, 0), (1, 1), (1, 0)])


def test__mask_by_threshold():
    result = geoprocessing._mask_by_threshold(A, 2)
    expected_result = np.array(
        [[False, False, True], [False, True, True], [True, True, True]]
    )

    np.testing.assert_array_equal(result, expected_result)


def test__mask_by_nodata():
    result = geoprocessing._mask_by_nodata(A, 2)
    expected_result = np.array(
        [[True, False, True], [False, True, True], [True, True, True]]
    )

    np.testing.assert_array_equal(result, expected_result)


@mock.patch("raster_analysis.geoprocessing.mask_geom_on_raster")
@mock.patch("raster_analysis.geoprocessing.read_window")
@mock.patch("raster_analysis.geoprocessing.read_window_ignore_missing")
def test_area_analysis(mock_data_ignore, mock_data, mock_masked_data):
    mock_masked_data.return_value = np.copy(A), np.copy(MASK), 0, None
    mock_data.side_effect = [(np.copy(B), None, None), (np.copy(C), None, None)]
    mock_data_ignore.side_effect = [(np.copy(B), None, None), (np.copy(C), None, None)]

    result = geoprocessing.analysis(
        GEOMETRY, "ras0", ["ras1", "ras2"], analyses=["area"]
    )
    expected_data = pd.DataFrame(
        {
            "ras0": [3, 4, 5],
            "ras1": [4, 5, 6],
            "ras2": [5, 6, 7],
            "area": [AREA * 3 / 10000, AREA * 2 / 10000, AREA / 10000],
        }
    )

    result_data = pd.DataFrame.from_dict(result["detailed_table"])

    for i, r in enumerate(result_data.area):
        assert r == pytest.approx(expected_data.area[i], 0.000001)


@mock.patch("raster_analysis.geoprocessing.mask_geom_on_raster")
@mock.patch("raster_analysis.geoprocessing.read_window")
@mock.patch("raster_analysis.geoprocessing.read_window_ignore_missing")
def test_sum_analysis(mock_data_ignore, mock_data, mock_masked_data):
    mock_masked_data.return_value = np.copy(A), np.copy(MASK), 0, None
    mock_data.side_effect = [(np.copy(F1), None, None), (np.copy(F2), None, None)]
    mock_data_ignore.side_effect = [
        (np.copy(F1), None, None),
        (np.copy(F2), None, None),
    ]

    result = geoprocessing.analysis(
        GEOMETRY, "ras0", aggregate_raster_ids=["ras1", "ras2"], analyses=["sum"]
    )
    expected_data = pd.DataFrame(
        {"ras0": [3, 4, 5], "ras1": [12.0, 10.0, 6.5], "ras2": [15.0, 12.0, 7.5]}
    )

    assert pd.DataFrame.from_dict(result["detailed_table"]).equals(expected_data)


@mock.patch("raster_analysis.geoprocessing.mask_geom_on_raster")
@mock.patch("raster_analysis.geoprocessing.read_window")
@mock.patch("raster_analysis.geoprocessing.read_window_ignore_missing")
def test_count_with_filter(mock_data_ignore, mock_data, mock_masked_data):
    mock_masked_data.return_value = np.copy(A), np.copy(MASK), 0, None
    mock_data.side_effect = [
        (FILTER, None, None),
        (np.copy(B), None, None),
        (np.copy(C), None, None),
    ]
    mock_data_ignore.side_effect = [(np.copy(B), None, None), (np.copy(C), None, None)]

    expected_data = pd.DataFrame(
        {"ras0": [3, 4, 5], "ras1": [4, 5, 6], "ras2": [5, 6, 7], "count": [2, 1, 1]}
    )

    result = geoprocessing.analysis(
        GEOMETRY,
        "ras0",
        ["ras1", "ras2"],
        filters=[Filter("filter", THRESHOLD)],
        analyses=["count"],
    )

    assert pd.DataFrame.from_dict(result["detailed_table"]).equals(expected_data)
