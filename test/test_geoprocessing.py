from raster_analysis import geoprocessing
from shapely.geometry import Polygon
from unittest import mock
import numpy as np
import affine
import pytest

A = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])

B = np.array([[2, 3, 4], [3, 4, 5], [4, 5, 6]])

C = np.array([[3, 4, 5], [4, 5, 6], [5, 6, 7]])

F1 = np.array([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])

F2 = np.array([[3.0, 4.0, 5.0], [4.0, 5.0, 6.0], [5.0, 6.0, 7.0]])

DT = [("A", "int"), ("B", "int"), ("C", "int")]
DT_F = [("A", "int"), ("F1", "float"), ("F2", "float")]
AREA_DT = [("A", "int"), ("B", "int"), ("C", "int"), ("AREA", "float")]
COUNT_DT = [("A", "int"), ("B", "int"), ("C", "int"), ("COUNT", "int")]

AREA = 769.288482

ARRAY = np.array(
    [(3, 4, 5), (3, 4, 5), (4, 5, 6), (3, 4, 5), (4, 5, 6), (5, 6, 7)], dtype=DT
)

AREA_ARRAY = np.array(
    [
        (3, 4, 5, AREA),
        (3, 4, 5, AREA),
        (4, 5, 6, AREA),
        (3, 4, 5, AREA),
        (4, 5, 6, AREA),
        (5, 6, 7, AREA),
    ],
    dtype=AREA_DT,
)

AREA_ARRAY2 = np.array(
    [
        (3, 4, 5, AREA, AREA * 2),
        (3, 4, 5, AREA, AREA * 2),
        (4, 5, 6, AREA, AREA * 2),
        (3, 4, 5, AREA, AREA * 2),
        (4, 5, 6, AREA, AREA * 2),
        (5, 6, 7, AREA, AREA * 2),
    ],
    dtype=AREA_DT + [("AREA2", "float")],
)

SUM_AREA_ARRAY = np.array(
    [(3, 4, 5, AREA * 3), (4, 5, 6, AREA * 2), (5, 6, 7, AREA)], dtype=AREA_DT
)

SUM_AREA_ARRAY2 = np.array(
    [
        (3, 4, 5, AREA * 3, AREA * 6),
        (4, 5, 6, AREA * 2, AREA * 4),
        (5, 6, 7, AREA, AREA * 2),
    ],
    dtype=AREA_DT + [("AREA2", "float")],
)

SUM_ARRAY = np.array([(3, 12.0, 15.0), (4, 8.0, 12.0), (5, 5.0, 7.0)], dtype=DT_F)

COUNT_ARRAY = np.array([(3, 4, 5, 3), (4, 5, 6, 2), (5, 6, 7, 1)], dtype=COUNT_DT)

MASK = np.array([[False, False, True], [False, True, True], [True, True, True]])

GEOMETRY = Polygon([(0, 0), (1, 1), (1, 0)])

EXTENT = GEOMETRY.bounds


class DummySrc(object):
    def __init__(self):
        self.nodata = 0
        self.transform = affine.Affine.identity()


class DummySrcA(DummySrc):
    def read(self, band, masked, window):
        if masked:
            return np.ma.array(A, mask=MASK)
        else:
            return A


class DummySrcB(DummySrc):
    def read(self, band, masked, window):
        if masked:
            return np.ma.array(B, mask=MASK)
        else:
            return B


class DummySrcC(DummySrc):
    def read(self, band, masked, window):
        if masked:
            return np.ma.array(C, mask=MASK)
        else:
            return C


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


def test__sum_area():
    result = geoprocessing._sum_area(ARRAY, AREA)
    expected_result = SUM_AREA_ARRAY
    np.testing.assert_array_equal(result, expected_result)


def test__sum():
    result = geoprocessing._sum(AREA_ARRAY)
    expected_result = SUM_AREA_ARRAY
    np.testing.assert_array_equal(result, expected_result)


def test__sum2():
    result = geoprocessing._sum(AREA_ARRAY2)
    expected_result = SUM_AREA_ARRAY2
    np.testing.assert_array_equal(result, expected_result)


def test__count():
    result = geoprocessing._count(ARRAY)
    expected_result = COUNT_ARRAY
    np.testing.assert_array_equal(result, expected_result)


@mock.patch("raster_analysis.geoprocessing.mask_geom_on_raster")
@mock.patch("raster_analysis.geoprocessing.read_window")
@mock.patch("raster_analysis.utilities.arrays.read_window_ignore_missing")
def test_area_analysis(mock_data_ignore, mock_data, mock_masked_data):
    mock_masked_data.return_value = np.ma.array(A, mask=MASK), 0, None
    mock_data.side_effect = [(B, None, None), (C, None, None)]
    mock_data_ignore.side_effect = [(B, None, None), (C, None, None)]

    result = geoprocessing.analysis(GEOMETRY, "ras0", "ras1", "ras2", analysis="area")

    expected_data = [
        (3, 4, 5, AREA * 3 / 10000),
        (4, 5, 6, AREA * 2 / 10000),
        (5, 6, 7, AREA / 10000),
    ]
    expected_dtype = [
        ("ras0", "<i8"),
        ("ras1", "<i8"),
        ("ras2", "<i8"),
        ("AREA", "<f8"),
    ]
    print(result)

    result_dtype = result["body"].pop("dtype")
    result_data = result["body"].pop("data")

    for i, r in enumerate(result_data):
        assert r == pytest.approx(expected_data[i], 0.000001)
    assert result_dtype == expected_dtype


@mock.patch("raster_analysis.geoprocessing.mask_geom_on_raster")
@mock.patch("raster_analysis.geoprocessing.read_window")
@mock.patch("raster_analysis.utilities.arrays.read_window_ignore_missing")
def test_sum_analysis(mock_data_ignore, mock_data, mock_masked_data):
    mock_masked_data.return_value = np.ma.array(A, mask=MASK), 0, None
    mock_data.side_effect = [(F1, None, None), (F2, None, None)]
    mock_data_ignore.side_effect = [(F1, None, None), (F2, None, None)]

    result = geoprocessing.analysis(GEOMETRY, "ras0", "ras1", "ras2", analysis="sum")
    expected_result = {
        "status": 200,
        "body": {
            "data": [(3, 12.0, 15.0), (4, 10.0, 12.0), (5, 6.0, 7.0)],
            "dtype": [("ras0", "<i8"), ("ras1", "<f8"), ("ras2", "<f8")],
        },
    }

    assert result == expected_result
