import threading
import uuid
import pytest
import subprocess
import os
from datetime import datetime, timedelta

from shapely.geometry import box, mapping

import raster_analysis.boto as boto
import raster_analysis
import lambdas.fanout.src.lambda_function
from lambdas.raster_analysis.src.lambda_function import handler as analysis_handler
from lambdas.tiled_analysis.src.lambda_function import handler as tiled_handler
import lambdas.tiled_analysis.src.lambda_function
from tests.fixtures.idn_24_9 import (
    IDN_24_9_GLAD_ALERTS,
    IDN_24_9_GEOM,
    IDN_24_9_GAIN,
    IDN_24_9_2010_EXTENT,
    IDN_24_9_LOSS_BY_DRIVER,
    IDN_24_9_PRIMARY_LOSS,
    IDN_24_9_ESA_LAND_COVER,
)

###
# TODO test Downloads/borneo_orangutan.zip and see what happens (prod geostore=fe14a1ec856d2a4888a7099b1a09e9aa)
###


class Context(object):
    def __init__(self, aws_request_id, log_stream_name):
        self.aws_request_id = aws_request_id
        self.log_stream_name = log_stream_name


@pytest.fixture(autouse=True)
def context(monkeypatch):
    def mock_lambda(payload, lambda_name, client):
        uid = str(uuid.uuid1())
        context = Context(uid, f"log_stream_{uid}")

        # don't import until here to makes sure monkey patch works
        from lambdas.fanout.src.lambda_function import handler as fanout_handler

        f = fanout_handler if lambda_name == "fanout" else analysis_handler
        thread = threading.Thread(target=f, args=(payload, context))
        thread.start()

    # monkey patch to just run on thread instead of actually invoking lambda
    monkeypatch.setattr(raster_analysis.tiling, "invoke_lambda", mock_lambda)
    monkeypatch.setattr(
        lambdas.fanout.src.lambda_function, "invoke_lambda", mock_lambda
    )

    os.environ["FANOUT_LAMBDA_NAME"] = "fanout"
    os.environ["RASTER_ANALYSIS_LAMBDA_NAME"] = "raster_analysis"
    os.environ["TILED_RESULTS_TABLE_NAME"] = "tiled-raster-analysis"
    os.environ[
        "S3_BUCKET_DATA_LAKE"
    ] = "gfw-data-lake"  # This is actual production data lake

    moto_server = subprocess.Popen(["moto_server", "dynamodb2", "-p3000"])
    try:
        boto.dynamodb_client().create_table(
            AttributeDefinitions=[
                {"AttributeName": "analysis_id", "AttributeType": "S"},
                {"AttributeName": "tile_id", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "analysis_id", "KeyType": "HASH"},
                {"AttributeName": "tile_id", "KeyType": "RANGE"},
            ],
            TableName="tiled-raster-analysis",
            BillingMode="PAY_PER_REQUEST",
        )

        uid = str(uuid.uuid1())
        context = Context(uid, f"log_stream_{uid}")
        yield context
    finally:
        moto_server.kill()


def test_primary_tree_cover_loss(context):
    result = tiled_handler(
        {
            "geometry": IDN_24_9_GEOM,
            "group_by": ["umd_tree_cover_loss__year"],
            "filters": [
                "is__umd_regional_primary_forest_2001",
                "umd_tree_cover_density_2000__30",
            ],
            "sum": ["area__ha", "whrc_aboveground_co2_emissions__Mg"],
        },
        context,
    )["body"]

    assert result["status"] == "success"
    for row_actual, row_expected in zip(result["data"], IDN_24_9_PRIMARY_LOSS):
        assert row_actual["area__ha"] == pytest.approx(row_expected["area__ha"], 0.001)
        assert row_actual["whrc_aboveground_co2_emissions__Mg"] == pytest.approx(
            row_expected["whrc_aboveground_co2_emissions__Mg"], 0.001
        )


def test_extent_2010(context):
    result = tiled_handler(
        {
            "geometry": IDN_24_9_GEOM,
            "filters": ["umd_tree_cover_density_2010__15"],
            "sum": ["area__ha"],
        },
        context,
    )["body"]

    assert result["status"] == "success"
    assert result["data"]["area__ha"] == pytest.approx(
        IDN_24_9_2010_EXTENT["area__ha"], 0.001
    )


def test_tree_cover_gain(context, monkeypatch):
    # let's also test encoded geometries
    monkeypatch.setattr(
        lambdas.tiled_analysis.src.lambda_function,
        "LAMBDA_ASYNC_PAYLOAD_LIMIT_BYTES",
        80000,
    )

    result = tiled_handler(
        {
            "geometry": IDN_24_9_GEOM,
            "filters": ["is__umd_tree_cover_gain"],
            "sum": ["area__ha"],
        },
        context,
    )["body"]

    assert result["status"] == "success"
    assert result["data"]["area__ha"] == pytest.approx(IDN_24_9_GAIN["area__ha"], 0.001)


def test_tree_cover_loss_by_driver(context):
    result = tiled_handler(
        {
            "geometry": IDN_24_9_GEOM,
            "group_by": [
                "umd_tree_cover_loss__year",
                "tsc_tree_cover_loss_drivers__type",
            ],
            "filters": ["umd_tree_cover_density_2000__30"],
            "sum": ["area__ha"],
        },
        context,
    )["body"]

    assert result["status"] == "success"
    for row_actual, row_expected in zip(result["data"], IDN_24_9_LOSS_BY_DRIVER):
        assert row_actual["area__ha"] == pytest.approx(row_expected["area__ha"], 0.001)


def test_glad_alerts(context):
    result = tiled_handler(
        {
            "geometry": IDN_24_9_GEOM,
            "group_by": ["umd_glad_alerts__isoweek"],
            "start_date": "2019-01-01",
            "end_date": "2019-12-31",
            "sum": ["alert__count"],
        },
        context,
    )["body"]

    assert result["status"] == "success"
    for row_actual, row_expected in zip(result["data"], IDN_24_9_GLAD_ALERTS):
        assert row_actual["alert__count"] == row_expected["alert__count"]


def test_land_cover_area(context):
    result = tiled_handler(
        {
            "geometry": IDN_24_9_GEOM,
            "group_by": ["esa_land_cover_2015__class"],
            "sum": ["area__ha"],
        },
        context,
    )["body"]

    assert result["status"] == "success"
    for row_actual, row_expected in zip(result["data"], IDN_24_9_ESA_LAND_COVER):
        print(
            f"{row_actual['esa_land_cover_2015__class']}, {row_expected['esa_land_cover_2015__class']}"
        )
        assert row_actual["area__ha"] == pytest.approx(row_expected["area__ha"], 0.001)


def test_error(context):
    start = datetime.now()
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "group_by": ["not_real"]}, context
    )["body"]
    end = datetime.now()

    timeout = timedelta(seconds=29)
    assert result["status"] == "failed"
    assert (end - start) < timeout


def test_beyond_extent(context):
    """
    Test a geometry outside the extent of is__umd_regional_primary_forest_2001
    """
    geometry = mapping(box(0, 40, 1, 41))
    result = tiled_handler(
        {
            "geometry": geometry,
            "group_by": ["umd_tree_cover_loss__year"],
            "filters": ["is__umd_regional_primary_forest_2001"],
            "sum": ["area__ha"],
        },
        context,
    )["body"]

    assert result["status"] == "success"
    assert not result["data"]
