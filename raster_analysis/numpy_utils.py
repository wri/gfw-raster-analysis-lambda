import numpy as np
from aws_xray_sdk.core import xray_recorder

from typing import Dict, List
from numpy import ndarray

DataFrame = Dict[str, ndarray]


@xray_recorder.capture("Group By")
def group_by(df: DataFrame, group_by_cols: List[str]) -> DataFrame:
    """
    Fast and low memory NumPy group by operation, since pandas group by was slow and very memory-intensive.
    :param df: Dict of column name -> NumPy array of column data
    :param group_by_cols: List of column names to group by; the rest will be summed based on groupings
    :return: Same format as input df but with grouped results
    """
    cols_1d = [np.ravel(df[col]) for col in group_by_cols]
    column_maxes = [col.max() + 1 for col in cols_1d]
    linear_indices = np.ravel_multi_index(cols_1d, column_maxes).astype(np.uint32)

    unique_values, inv = np.unique(linear_indices, return_inverse=True)
    unique_value_combinations = np.unravel_index(unique_values, column_maxes)

    results = dict(zip(df.keys(), unique_value_combinations))

    for col, data in df.items():
        if col not in group_by_cols:
            results[col] = np.bincount(inv, weights=data).astype(data.dtype)

    return results


@xray_recorder.capture("Get Linear Index")
def get_linear_index(cols, dims, mask=None):
    linear_index = np.ravel_multi_index(cols, dims).astype(np.uint32)
    if mask is not None:
        linear_index = np.compress(np.ravel(mask), linear_index)

    return linear_index
