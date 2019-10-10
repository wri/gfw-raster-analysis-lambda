from unittest import mock
from shapely.geometry import shape
from raster_analysis import geoprocessing
from raster_analysis.geoprocessing import Filter
from test.generate_test_data import generate_test_data
import json
import os
import os.path

TEST_DATA_FOLDER = os.path.abspath("test_data")
TEST_JSON_RESULTS = "test_geom_results.json"

if not os.path.exists("test_data"):
    generate_test_data()


def get_path_by_id(id):
    return os.path.join(TEST_DATA_FOLDER, id) + ".tif"


@mock.patch("raster_analysis.geoprocessing.get_raster_url")
def test_basic(mock_raster_url):
    mock_raster_url.side_effect = get_path_by_id

    with open(TEST_JSON_RESULTS, "r") as f:
        test_data = json.load(f)["basic"]

    geometry = shape(test_data["geometry"])
    results = geoprocessing.analysis(
        geometry,
        "analysis_layer",
        ["contextual_layer_1", "contextual_layer_2"],
        ["sum_layer"],
        [Filter("filter_layer", 8)],
        ["count", "sum"],
    )

    assert json.dumps(test_data["results"]["detailed_table"]) == json.dumps(
        results["detailed_table"]
    )


@mock.patch("raster_analysis.geoprocessing.get_raster_url")
def test_geometry_with_hole(mock_raster_url):
    mock_raster_url.side_effect = get_path_by_id

    with open(TEST_JSON_RESULTS, "r") as f:
        test_data = json.load(f)["geom_with_hole"]

    geometry = shape(test_data["geometry"])
    results = geoprocessing.analysis(
        geometry,
        "analysis_layer",
        ["contextual_layer_1", "contextual_layer_2"],
        [],
        [Filter("filter_layer", 8)],
        ["count"],
    )

    assert json.dumps(test_data["results"]["detailed_table"]) == json.dumps(
        results["detailed_table"]
    )
