from mock import patch, call
import pytest
import os
from shapely.geometry import box

os.environ["RASTER_ANALYSIS_LAMBDA_NAME"] = "test_raster_analysis"

from raster_analysis.tiling import (
    get_intersecting_geoms,
    get_tiles,
    merge_tile_results,
    process_tiled_geoms,
    _get_rounded_bounding_box,
)


@patch("raster_analysis.tiling.run_raster_analysis")
def test_process_tiled_geoms(mock_raster_analysis):
    tile_results = [
        {
            "ras0": [1, 1, 2, 4, 4, 7, 7, 9],
            "ras1": [0, 1, 1, 0, 1, 0, 1, 0],
            "cont": [3, 8, 4, 2, 9, 4, 5, 8],
        },
        {
            "ras0": [1, 2, 2, 3, 3, 6, 7, 8, 9, 9, 10],
            "ras1": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            "cont": [6, 2, 1, 4, 3, 2, 9, 8, 4, 1, 6],
        },
        {"ras0": [], "ras1": [], "cont": []},
    ]

    tiled_geoms = get_tiles(box(0.2, 0.2, 1.0, 2.4), 1)
    geoprocessing_params = {"analysis_raster_id": "loss"}

    mock_raster_analysis.side_effect = tile_results

    actual_results = process_tiled_geoms(tiled_geoms, geoprocessing_params)

    assert mock_raster_analysis.call_count == 3
    assert len(actual_results) == 2
    assert actual_results[0] == tile_results[0]
    assert actual_results[1] == tile_results[1]


def test_merge_tile_results():
    tile_results = [
        {
            "ras0": [1, 1, 2, 4, 4, 7, 7, 9],
            "ras1": [0, 1, 1, 0, 1, 0, 1, 0],
            "cont": [3, 8, 4, 2, 9, 4, 5, 8],
            "ras2": [3.2, 8.2, 4.2, 2.2, 9.2, 4.2, 5.2, 8.2],
        },
        {
            "ras0": [1, 2, 2, 3, 3, 6, 7, 8, 9, 9, 10],
            "ras1": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            "cont": [6, 2, 1, 4, 3, 2, 9, 8, 4, 1, 6],
            "ras2": [6.2, 2.2, 1.2, 4.2, 3.2, 2.2, 9.2, 8.2, 4.2, 1.2, 6.2],
        },
    ]

    results = merge_tile_results(tile_results, ["ras0", "ras1"])
    assert results["ras0"] == [1, 1, 2, 2, 3, 3, 4, 4, 6, 7, 7, 8, 9, 9, 10]
    assert results["ras1"] == [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    assert results["cont"] == [9, 8, 2, 5, 4, 3, 2, 9, 2, 4, 14, 8, 12, 1, 6]

    ras2_result = [
        9.4,
        8.2,
        2.2,
        5.4,
        4.2,
        3.2,
        2.2,
        9.2,
        2.2,
        4.2,
        14.4,
        8.2,
        12.4,
        1.2,
        6.2,
    ]
    approx_result = [
        pytest.approx(n, 0.00001) for n in ras2_result
    ]  # deal with floating arithmetic imprecision

    assert results["ras2"] == approx_result


def test_get_intersecting_geoms():
    geom = box(0.2, 0.2, 1.4, 1.4)
    tiles = get_tiles(geom, 1)
    inter_geoms = get_intersecting_geoms(geom, tiles)

    assert len(inter_geoms) == 4
    assert any([g.bounds == (0.2, 0.2, 1.0, 1.0) for g in inter_geoms])
    assert any([g.bounds == (1.0, 1.0, 1.4, 1.4) for g in inter_geoms])
    assert any([g.bounds == (0.2, 1.0, 1.0, 1.4) for g in inter_geoms])
    assert any([g.bounds == (1.0, 0.2, 1.4, 1.0) for g in inter_geoms])


def test_get_tiles():
    geom = box(0.2, 0.2, 1.4, 1.4)
    tiles = get_tiles(geom, 1)

    assert len(tiles) == 4
    assert any([t == box(0, 0, 1, 1) for t in tiles])
    assert any([t == box(0, 1, 1, 2) for t in tiles])
    assert any([t == box(1, 0, 2, 1) for t in tiles])
    assert any([t == box(1, 1, 2, 2) for t in tiles])


def test_get_rounded_bounding_box():
    geom = box(0.2, 0.3, 1.3, 1.4)
    rounded_bbox = _get_rounded_bounding_box(geom, 1.0)

    assert rounded_bbox == (0.0, 0.0, 2.0, 2.0)
