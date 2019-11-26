from raster_analysis.geoprocessing import RasterWindow, _analysis
import pandas as pd
import numpy as np


def test_analysis():
    # stack two arrays with values 1 and 2
    ones = np.ones((5, 10), dtype=np.int64)
    twos = ones * 2
    test_arr1 = np.vstack([ones, twos])
    test_arr2 = test_arr1.transpose()

    sub_mask = np.ones((6, 6), dtype=np.int64)
    test_mask = np.zeros((10, 10), dtype=np.int64)
    test_mask[2:8, 2:8] += sub_mask

    test_sum_arr = np.ones((10, 10)) * 0.5

    raster_windows = {
        "arr1": RasterWindow(test_arr1, None, None),
        "arr2": RasterWindow(test_arr2, None, None),
        "sum_arr": RasterWindow(test_sum_arr, None, None),
    }

    result = _analysis(
        ["count", "area"], raster_windows, ["arr1", "arr2"], ["sum_arr"], 2, test_mask
    )

    assert result["count"].sum() == 9 * 4
    assert result["area"].sum() == 18 * 4
    assert result["sum_arr"].sum() == 4.5 * 4
    assert (result["arr1"] == [1, 1, 2, 2]).sum() == 4
    assert (result["arr2"] == [1, 2, 1, 2]).sum() == 4
    assert (result["count"] == [9, 9, 9, 9]).sum() == 4
    assert (result["sum_arr"] == [4.5, 4.5, 4.5, 4.5]).sum() == 4
