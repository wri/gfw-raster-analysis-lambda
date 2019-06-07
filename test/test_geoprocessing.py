from raster_analysis import geoprocessing
from shapely.geometry import Polygon
from unittest import mock
import numpy as np
import affine

A = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5]])

B = np.array([[2, 3, 4],
              [3, 4, 5],
              [4, 5, 6]])

C = np.array([[3, 4, 5],
              [4, 5, 6],
              [5, 6, 7]])

F1 = np.array([[2., 3., 4.],
               [3., 4., 5.],
               [4., 5., 6.]])

F2 = np.array([[3., 4., 5.],
               [4., 5., 6.],
               [5., 6., 7.]])

DT = [("A", "int"), ("B", "int"), ("C", "int")]
DT_F = [("A", "int"), ("F1", "float"), ("F2", "float")]
AREA_DT = [("A", "int"), ("B", "int"), ("C", "int"), ("AREA", "float")]
COUNT_DT = [("A", "int"), ("B", "int"), ("C", "int"), ("COUNT", "int")]

AREA = 769.288482

ARRAY = np.array([(3, 4, 5),
                  (3, 4, 5),
                  (4, 5, 6),
                  (3, 4, 5),
                  (4, 5, 6),
                  (5, 6, 7)], dtype=DT)

AREA_ARRAY = np.array([(3, 4, 5, AREA),
                       (3, 4, 5, AREA),
                       (4, 5, 6, AREA),
                       (3, 4, 5, AREA),
                       (4, 5, 6, AREA),
                       (5, 6, 7, AREA)], dtype=AREA_DT)

AREA_ARRAY2 = np.array([(3, 4, 5, AREA, AREA * 2),
                        (3, 4, 5, AREA, AREA * 2),
                        (4, 5, 6, AREA, AREA * 2),
                        (3, 4, 5, AREA, AREA * 2),
                        (4, 5, 6, AREA, AREA * 2),
                        (5, 6, 7, AREA, AREA * 2)], dtype=AREA_DT + [("AREA2", "float")])

SUM_AREA_ARRAY = np.array([(3, 4, 5, AREA * 3),
                           (4, 5, 6, AREA * 2),
                           (5, 6, 7, AREA)], dtype=AREA_DT)

SUM_AREA_ARRAY2 = np.array([(3, 4, 5, AREA * 3, AREA * 6),
                            (4, 5, 6, AREA * 2, AREA * 4),
                            (5, 6, 7, AREA, AREA * 2)], dtype=AREA_DT + [("AREA2", "float")])

SUM_ARRAY = np.array([(3, 12., 15.),
                      (4, 8., 12.),
                      (5, 5., 7.)], dtype=DT_F)

COUNT_ARRAY = np.array([(3, 4, 5, 3),
                        (4, 5, 6, 2),
                        (5, 6, 7, 1)], dtype=COUNT_DT)

MASK = np.array([[False, False, True],
                 [False, True, True],
                 [True, True, True]])

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
    expected_result = np.array([[False, False, True],
                                [False, True, True],
                                [True, True, True]])

    np.testing.assert_array_equal(result, expected_result)


def test__mask_by_nodata():
    result = geoprocessing._mask_by_nodata(A, 2)
    expected_result = np.array([[True, False, True],
                                [False, True, True],
                                [True, True, True]])

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
def test_sum_analysis_area(mock_data_ignore, mock_data, mock_masked_data):
    mock_masked_data.return_value = np.ma.array(A, mask=MASK), 0, None
    mock_data.side_effect = [(B, None, None), (C, None, None)]
    mock_data_ignore.side_effect = [(B, None, None), (C, None, None)]

    result = geoprocessing.sum_analysis(GEOMETRY, "ras0", "ras1", "ras2", area=True)

    expected_result = {"data": [(3, 4, 5, 2307.8654454620364),
                                (4, 5, 6, 1538.5769636413575),
                                (5, 6, 7, 769.2884818206787)],
                       "extent_2000": None,
                       "extent_2010": None}
    print(result)
    assert result == expected_result


@mock.patch("raster_analysis.geoprocessing.mask_geom_on_raster")
@mock.patch("raster_analysis.geoprocessing.read_window")
@mock.patch("raster_analysis.utilities.arrays.read_window_ignore_missing")
def test_sum_analysis(mock_data_ignore, mock_data, mock_masked_data):
    mock_masked_data.return_value = np.ma.array(A, mask=MASK), 0, None
    mock_data.side_effect = [(F1, None, None), (F2, None, None)]
    mock_data_ignore.side_effect = [(F1, None, None), (F2, None, None)]

    result = geoprocessing.sum_analysis(GEOMETRY, "ras0", "ras1", "ras2", area=False)

    expected_result = {"data": [(3, 12., 15.),
                                (4, 10., 12.),
                                (5, 6., 7.)],
                       "extent_2000": None,
                       "extent_2010": None}
    print(result)
    assert result == expected_result
