from raster_analysis import arrays
from unittest import mock
from shapely.geometry import Polygon
import numpy as np

A = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])

B = np.array([[2, 3, 4], [3, 4, 5], [4, 5, 6]])

C = np.array([[3, 4, 5], [4, 5, 6], [5, 6, 7]])

MASK = np.array([[False, False, True], [False, True, True], [True, True, True]])

GEOMETRY = Polygon([(0, 0), (1, 1), (1, 0)])

STRUCTURED_ARRAY = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=[("test", "int")])
STRUCTURED_ARRAY_BOOL = np.array([True, False, True], dtype=[("bool", "bool_")])
STRUCTURED_ARRAY_INT = np.array([1, 2, 3], dtype=[("int", "int")])
STRUCTURED_ARRAY_FLOAT = np.array([1.0, 2.0, 3.0], dtype=[("float", "float")])

COMBINED_ARRAY = np.array(
    [(3, 4, 5), (3, 4, 5), (4, 5, 6), (3, 4, 5), (4, 5, 6), (5, 6, 7)],
    dtype=[("test", "<i8"), ("src_b", "<i8"), ("src_c", "<i8")],
)

COMBINED_VIEW = np.array(
    [(4, 5), (4, 5), (5, 6), (4, 5), (5, 6), (6, 7)],
    dtype=[("src_b", "<i8"), ("src_c", "<i8")],
)

DT = np.dtype([("A", "int"), ("B", "int"), ("C", "int"), ("AREA", "float")])


def test_to_structured_array():
    result = arrays.to_structured_array(A, "test")
    expected_result = STRUCTURED_ARRAY

    np.testing.assert_array_equal(result, expected_result)


def test__dtype_to_list():
    result = arrays.dtype_to_list(STRUCTURED_ARRAY.dtype)
    expected_result = [("test", np.dtype("int64"))]

    assert result == expected_result


def test__fill_array():
    dt = [("bool", "bool_"), ("int", "int"), ("float", "float")]
    e = np.empty(3, dtype=dt)
    result = arrays.fill_array(
        e, STRUCTURED_ARRAY_BOOL, STRUCTURED_ARRAY_INT, STRUCTURED_ARRAY_FLOAT
    )
    expected_result = np.array(
        [(True, 1, 1.0), (False, 2, 2.0), (True, 3, 3.0)], dtype=dt
    )

    np.testing.assert_array_equal(result, expected_result)


def test__build_array():
    dt = [("bool", "bool_"), ("int", "int")]
    result = arrays.concat_arrays(STRUCTURED_ARRAY_BOOL, STRUCTURED_ARRAY_INT)
    expected_result = np.array([(True, 1), (False, 2), (True, 3)], dtype=dt)

    np.testing.assert_array_equal(result, expected_result)


@mock.patch("raster_analysis.arrays.read_window_ignore_missing")
def test_build_array(mock_data):
    mock_data.side_effect = [(B, None, None), (C, None, None)]
    result = arrays.build_array(MASK, STRUCTURED_ARRAY, "src_b", "src_c", geom=GEOMETRY)
    expected_result = COMBINED_ARRAY

    np.testing.assert_array_equal(result, expected_result)


def test_field_view():
    result = arrays.fields_view(COMBINED_ARRAY, ["src_b", "src_c"])
    expected_result = COMBINED_VIEW

    np.testing.assert_array_equal(result, expected_result)


def test_get_fields_by_type_include():
    result = arrays.get_fields_by_type(DT, "int")
    expected_results = ["A", "B", "C"]

    assert result == expected_results


def test_get_fields_by_type_exclude():
    result = arrays.get_fields_by_type(DT, "int", exclude=True)
    expected_results = ["AREA"]

    assert result == expected_results
