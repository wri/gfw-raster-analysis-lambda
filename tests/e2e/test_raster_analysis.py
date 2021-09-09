# flake8: noqa
import os
import subprocess
import uuid
from datetime import datetime, timedelta
from threading import Thread

import pytest
from shapely.geometry import box, mapping

# set environment before importing our lambda layer
os.environ["FANOUT_LAMBDA_NAME"] = "fanout"
os.environ["RASTER_ANALYSIS_LAMBDA_NAME"] = "raster_analysis"
os.environ["TILED_RESULTS_TABLE_NAME"] = "tiled-raster-analysis"
os.environ["TILED_STATUS_TABLE_NAME"] = "tiled-raster-analysis-status"
os.environ["DYNAMODB_ENDPOINT_URL"] = "http://127.0.0.1:3000"
os.environ[
    "S3_BUCKET_DATA_LAKE"
] = "gfw-data-lake"  # This is actual production data lake

import lambdas.fanout.src.lambda_function
import lambdas.tiled_analysis.src.lambda_function
import raster_analysis
import raster_analysis.boto as boto
from lambdas.raster_analysis.src.lambda_function import handler as analysis_handler
from lambdas.tiled_analysis.src.lambda_function import handler as tiled_handler
from tests.fixtures.fixtures import (
    BRA_14_87_GEOM,
    COD_21_4_GEOM,
    DATA_ENVIRONMENT,
    IDN_24_9_2010_EXTENT,
    IDN_24_9_2010_RAW_AREA,
    IDN_24_9_2019_GLAD_ALERTS_TOTAL,
    IDN_24_9_ESA_LAND_COVER,
    IDN_24_9_GAIN,
    IDN_24_9_GEOM,
    IDN_24_9_GLAD_ALERTS,
    IDN_24_9_LOSS_BY_DRIVER,
    IDN_24_9_NET_FLUX,
    IDN_24_9_PRIMARY_LOSS,
)


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
        p = Thread(target=f, args=(payload, context))
        p.start()

    # monkey patch to just run on thread instead of actually invoking lambda
    monkeypatch.setattr(raster_analysis.tiling, "invoke_lambda", mock_lambda)
    monkeypatch.setattr(
        lambdas.fanout.src.lambda_function, "invoke_lambda", mock_lambda
    )

    moto_server = subprocess.Popen(["moto_server", "dynamodb2", "-p3000"])
    try:
        boto.dynamodb_client().create_table(
            AttributeDefinitions=[
                {"AttributeName": "tile_id", "AttributeType": "S"},
                {"AttributeName": "part_id", "AttributeType": "N"},
            ],
            KeySchema=[
                {"AttributeName": "tile_id", "KeyType": "HASH"},
                {"AttributeName": "part_id", "KeyType": "RANGE"},
            ],
            TableName="tiled-raster-analysis",
            BillingMode="PAY_PER_REQUEST",
        )

        boto.dynamodb_client().create_table(
            AttributeDefinitions=[
                {"AttributeName": "tile_id", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "tile_id", "KeyType": "HASH"},
            ],
            TableName="tiled-raster-analysis-status",
            BillingMode="PAY_PER_REQUEST",
        )

        uid = str(uuid.uuid1())
        context = Context(uid, f"log_stream_{uid}")
        yield context
    finally:
        moto_server.kill()


def test_primary_tree_cover_loss(context):
    query = "select sum(area__ha), sum(gfw_forest_carbon_gross_emissions__Mg_CO2e) from umd_tree_cover_loss__year where is__umd_regional_primary_forest_2001 = 'true' and (umd_tree_cover_density_2000__threshold >= 30 or is__umd_tree_cover_gain = 'true') group by umd_tree_cover_loss__year"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"
    assert result["data"]
    for row_actual, row_expected in zip(result["data"], IDN_24_9_PRIMARY_LOSS):
        assert row_actual["area__ha"] == pytest.approx(row_expected["area__ha"], 0.01)
        assert row_actual[
            "gfw_forest_carbon_gross_emissions__Mg_CO2e"
        ] == pytest.approx(
            row_expected["gfw_forest_carbon_gross_emissions__Mg_CO2e"], 0.01
        )


def test_extent_2010(context):
    query = "select sum(area__ha) from umd_tree_cover_density_2000__threshold where umd_tree_cover_density_2000__threshold >= 15"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )
    assert result["status"] == "success"

    assert result["data"][0]["area__ha"] == pytest.approx(
        IDN_24_9_2010_EXTENT["area__ha"], 0.01
    )


def test_lat_lon(context):
    query = "select latitude, longitude, umd_glad_landsat_alerts__date, umd_glad_landsat_alerts__confidence, is__umd_regional_primary_forest_2001 from umd_glad_landsat_alerts__date where umd_glad_landsat_alerts__date >= '2019-01-01' and umd_glad_landsat_alerts__date < '2020-01-01'"
    result = tiled_handler(
        {
            "geometry": IDN_24_9_GEOM,
            "query": query,
            "format": "csv",
            "environment": DATA_ENVIRONMENT,
        },
        context,
    )
    assert result["status"] == "success"

    lines = result["data"].splitlines()
    assert len(lines) == IDN_24_9_2019_GLAD_ALERTS_TOTAL


def test_raw_area(context):
    query = "select sum(area__ha) from data"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"
    assert result["data"][0]["area__ha"] == pytest.approx(
        IDN_24_9_2010_RAW_AREA["area__ha"], 0.001
    )


def test_tree_cover_gain(context, monkeypatch):
    # let's also test encoded geometries
    monkeypatch.setattr(
        raster_analysis.tiling, "LAMBDA_ASYNC_PAYLOAD_LIMIT_BYTES", 80000
    )

    query = "select sum(area__ha) from is__umd_tree_cover_gain"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"
    assert result["data"][0]["area__ha"] == pytest.approx(
        IDN_24_9_GAIN["area__ha"], 0.001
    )


def test_tree_cover_loss_by_driver(context):
    query = "select sum(area__ha) from umd_tree_cover_loss__year where umd_tree_cover_density_2000__threshold >= 30 group by umd_tree_cover_loss__year, tsc_tree_cover_loss_drivers__type"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"
    for row_actual, row_expected in zip(result["data"], IDN_24_9_LOSS_BY_DRIVER):
        assert row_actual["area__ha"] == pytest.approx(row_expected["area__ha"], 0.01)


def test_glad_alerts(context):
    query = "select count(umd_glad_landsat_alerts) from umd_glad_landsat_alerts__date where umd_glad_landsat_alerts__date >= '2019-01-01' and umd_glad_landsat_alerts__date < '2020-01-01' group by isoweek(umd_glad_landsat_alerts__date)"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"
    for row_actual, row_expected in zip(result["data"], IDN_24_9_GLAD_ALERTS):
        assert row_actual["count"] == row_expected["alert__count"]


def test_glad_alerts_count(context):
    query = "select count(umd_glad_landsat_alerts) from umd_glad_landsat_alerts__date where umd_glad_landsat_alerts__date >= '2019-01-01' and umd_glad_landsat_alerts__date < '2020-01-01' group by isoweek(umd_glad_landsat_alerts__date)"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"
    for row_actual, row_expected in zip(result["data"], IDN_24_9_GLAD_ALERTS):
        assert row_actual["count"] == row_expected["alert__count"]


def test_radd_alerts(context):
    # TODO calculate number of alerts offline
    query = "select latitude, longitude, gfw_radd_alerts__date, gfw_radd_alerts__confidence from gfw_radd_alerts__date where is__umd_regional_primary_forest_2001 = 'true' and gfw_radd_alerts__date >= '2021-01-01'"
    result = tiled_handler(
        {"geometry": COD_21_4_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"


def test_glad_s2_alerts(context):
    # TODO calculate number of alerts offline
    query = "select latitude, longitude, umd_glad_sentinel2_alerts__date, umd_glad_sentinel2_alerts__confidence from umd_glad_sentinel2_alerts__date where is__umd_regional_primary_forest_2001 = 'true' and umd_glad_sentinel2_alerts__date >= '2021-03-01'"
    result = tiled_handler(
        {"geometry": BRA_14_87_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"


def test_land_cover_area(context):
    query = "select sum(area__ha) from esa_land_cover_2015__class group by esa_land_cover_2015__class"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"
    for row_actual, row_expected in zip(result["data"], IDN_24_9_ESA_LAND_COVER):
        print(
            f"{row_actual['esa_land_cover_2015__class']}, {row_expected['esa_land_cover_2015__class']}"
        )
        assert row_actual["area__ha"] == pytest.approx(row_expected["area__ha"], 0.001)


def test_failed(context):
    start = datetime.now()
    query = "select sum(area__ha) from incorrect group by not_real"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )
    end = datetime.now()

    timeout = timedelta(seconds=29)
    assert result["status"] == "failed"
    assert (end - start) < timeout


def test_beyond_extent(context):
    """Test a geometry outside the extent of
    is__umd_regional_primary_forest_2001."""
    geometry = mapping(box(0, 40, 1, 41))
    query = "select sum(area__ha) from is__umd_regional_primary_forest_2001 group by umd_tree_cover_loss__year"
    result = tiled_handler(
        {"geometry": geometry, "query": query, "environment": DATA_ENVIRONMENT}, context
    )

    assert result["status"] == "success"
    assert not result["data"]


def test_net_flux(context):
    query = "select sum(gfw_forest_carbon_net_flux__Mg_CO2e), sum(gfw_forest_carbon_gross_emissions__Mg_CO2e), sum(gfw_forest_carbon_gross_removals__Mg_CO2e) from data where umd_tree_cover_density_2000__threshold >= 30 or is__umd_tree_cover_gain = 'true'"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"
    assert result["data"][0]["gfw_forest_carbon_net_flux__Mg_CO2e"] == pytest.approx(
        IDN_24_9_NET_FLUX["gfw_forest_carbon_net_flux__Mg_CO2e"], 0.01
    )
    assert result["data"][0][
        "gfw_forest_carbon_gross_emissions__Mg_CO2e"
    ] == pytest.approx(
        IDN_24_9_NET_FLUX["gfw_forest_carbon_gross_emissions__Mg_CO2e"], 0.01
    )
    assert result["data"][0][
        "gfw_forest_carbon_gross_removals__Mg_CO2e"
    ] == pytest.approx(
        IDN_24_9_NET_FLUX["gfw_forest_carbon_gross_removals__Mg_CO2e"], 0.01
    )
