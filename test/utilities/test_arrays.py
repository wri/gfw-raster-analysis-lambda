from raster_analysis.utilities import arrays
from unittest import mock
from shapely.geometry import Polygon
import numpy as np


A = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5]])

B = np.array([[2, 3, 4],
              [3, 4, 5],
              [4, 5, 6]])

C = np.array([[3, 4, 5],
              [4, 5, 6],
              [5, 6, 7]])

MASK = np.array([[False, False, True],
                 [False, True, True],
                 [True, True, True]])

GEOMETRY = Polygon([(0, 0), (1, 1), (1, 0)])

ARRAY = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]])
STRUCTURED_ARRAY = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]], dtype=[("test", "int")])
STRUCTURED_ARRAY_BOOL = np.array([True, False, True], dtype=[("bool", "bool_")])
STRUCTURED_ARRAY_INT = np.array([1, 2, 3], dtype=[("int", "int")])
STRUCTURED_ARRAY_FLOAT = np.array([1., 2., 3.], dtype=[("float", "float")])


def test_to_structured_array():
    result = arrays.to_structured_array(ARRAY, "test")
    expected_result = STRUCTURED_ARRAY

    np.testing.assert_array_equal(result, expected_result)


def test__dtype_to_list():
    result = arrays._dtype_to_list(STRUCTURED_ARRAY.dtype)
    expected_result = [("test", np.dtype("int64"))]

    assert result == expected_result


def test__fill_array():
    dt = ([("bool", "bool_"), ("int", "int"), ("float", "float")])
    e = np.empty(3, dtype=dt)
    result = arrays._fill_array(e, STRUCTURED_ARRAY_BOOL, STRUCTURED_ARRAY_INT, STRUCTURED_ARRAY_FLOAT)
    expected_result = np.array([(True, 1, 1.), (False, 2, 2.), (True, 3, 3.)], dtype=dt)

    np.testing.assert_array_equal(result, expected_result)


def test__build_array():
    dt = ([("bool", "bool_"), ("int", "int")])
    result = arrays._build_array(STRUCTURED_ARRAY_BOOL, STRUCTURED_ARRAY_INT)
    expected_result = np.array([(True, 1), (False, 2), (True, 3)], dtype=dt)

    np.testing.assert_array_equal(result, expected_result)


@mock.patch("raster_analysis.geoprocessing.read_window_ignore_missing")
def test_build_array(mock_data):
    mock_data.side_effect = [(B, None, None), (C, None, None)]
    result = arrays.build_array(MASK, A, "src_b", "src_c", geom=GEOMETRY)
    expected_result = ARRAY

    np.testing.assert_array_equal(result, expected_result)


