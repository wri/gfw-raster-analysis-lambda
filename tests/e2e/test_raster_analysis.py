import os
import socket
import subprocess
import sys
import time
import traceback
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from queue import Queue
from threading import Thread

import pytest
from shapely.geometry import box, mapping

import lambdas.fanout.src.lambda_function

# set environment before importing our lambda layer
os.environ["FANOUT_LAMBDA_NAME"] = "fanout"
os.environ["RASTER_ANALYSIS_LAMBDA_NAME"] = "raster_analysis"
os.environ["TILED_RESULTS_TABLE_NAME"] = "tiled-raster-analysis"
os.environ["TILED_STATUS_TABLE_NAME"] = "tiled-raster-analysis-status"
os.environ["DYNAMODB_ENDPOINT_URL"] = "http://127.0.0.1:3000"
os.environ["RESULTS_CHECK_TIMEOUT"] = "120"  # extend timeout in case local wifi is slow
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
    COD_24_1_TCLF,
    DATA_ENVIRONMENT,
    DATA_ENVIRONMENT_10M,
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

"""
This is a drop-in replacement for the context fixture in test_raster_analysis.py
that adds proper thread exception handling and moto server management.
"""

class Context(object):
    """Mock AWS Lambda context object."""

    def __init__(self, aws_request_id, log_stream_name):
        self.aws_request_id = aws_request_id
        self.log_stream_name = log_stream_name


def wait_for_port(host, port, timeout=10):
    """
    Wait for a TCP port to become available.

    Args:
        host: Hostname or IP address
        port: Port number
        timeout: Maximum time to wait in seconds

    Returns:
        bool: True if port is available, False if timeout
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (socket.error, ConnectionRefusedError, OSError):
            time.sleep(0.1)
    return False


@pytest.fixture()
def context(monkeypatch):
    """
    Fixture that sets up the test environment with improved error handling.

    Key improvements:
    1. Thread exception capture and reporting
    2. Moto server startup verification
    3. Better subprocess output capture
    4. Proper cleanup even on failure
    """

    # Queue to capture exceptions from threads
    exceptions_queue = Queue()
    threads_list = []

    def mock_lambda(payload, lambda_name, client):
        """
        Mock Lambda invocation by running handler in a thread.

        This wrapper captures exceptions and provides better error reporting.
        """
        uid = str(uuid.uuid1())
        context_obj = Context(uid, f"log_stream_{uid}")

        # Import handler based on lambda name
        from lambdas.fanout.src.lambda_function import handler as fanout_handler

        handler_func = fanout_handler if lambda_name == "fanout" else analysis_handler

        def wrapped_target():
            """Target function that captures exceptions."""
            try:
                print(f"[THREAD {lambda_name}] Starting handler execution", file=sys.stderr)
                result = handler_func(payload, context_obj)
                print(f"[THREAD {lambda_name}] Handler completed successfully", file=sys.stderr)
                return result
            except Exception as e:
                # Capture the full exception with traceback
                exc_info = sys.exc_info()
                exceptions_queue.put((lambda_name, exc_info))

                # Print to stderr so it shows up in pytest output immediately
                print(f"\n{'=' * 70}", file=sys.stderr)
                print(f"EXCEPTION IN THREAD ({lambda_name}):", file=sys.stderr)
                print(f"Exception Type: {type(e).__name__}", file=sys.stderr)
                print(f"Exception Value: {e}", file=sys.stderr)
                print(f"{'=' * 70}", file=sys.stderr)
                print(''.join(traceback.format_exception(*exc_info)), file=sys.stderr)
                print(f"{'=' * 70}\n", file=sys.stderr)

                # Re-raise so thread terminates with error
                raise

        # Create and start thread
        thread = Thread(
            target=wrapped_target,
            name=f"{lambda_name}-{uid[:8]}",
            daemon=False  # Non-daemon so we can track completion
        )
        thread.start()
        threads_list.append((lambda_name, thread))

        print(f"[MAIN] Started thread for {lambda_name}", file=sys.stderr)

    # Monkey patch to use our mock instead of real Lambda invocation
    monkeypatch.setattr(raster_analysis.tiling, "invoke_lambda", mock_lambda)
    monkeypatch.setattr(
        lambdas.fanout.src.lambda_function, "invoke_lambda", mock_lambda
    )

    print("\n[SETUP] Starting moto server...", file=sys.stderr)

    # Start moto server with output capture
    moto_server = subprocess.Popen(
        ["moto_server", "dynamodb", "-p3000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        # Wait for server to be ready
        print("[SETUP] Waiting for moto server to be ready...", file=sys.stderr)
        if not wait_for_port("127.0.0.1", 3000, timeout=10):
            # Server failed to start - try to get output
            moto_server.terminate()
            try:
                stdout, stderr = moto_server.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                moto_server.kill()
                stdout, stderr = moto_server.communicate()

            raise RuntimeError(
                f"Moto server failed to start on port 3000.\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}\n"
                f"Check if port 3000 is already in use: lsof -i :3000"
            )

        print("[SETUP] Moto server is ready, creating tables...", file=sys.stderr)

        # Give it a moment to fully initialize
        time.sleep(0.5)

        # Create DynamoDB tables
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
            print("[SETUP] DynamoDB tables created successfully", file=sys.stderr)
        except Exception as e:
            print(f"[SETUP] Failed to create DynamoDB tables: {e}", file=sys.stderr)
            raise

        # Create context object
        uid = str(uuid.uuid1())
        context_obj = Context(uid, f"log_stream_{uid}")

        print("[SETUP] Test environment ready\n", file=sys.stderr)

        # Yield to run the test
        yield context_obj

    finally:
        print("\n[TEARDOWN] Starting cleanup...", file=sys.stderr)

        # Wait for all threads to complete (with timeout)
        print(f"[TEARDOWN] Waiting for {len(threads_list)} threads to complete...", file=sys.stderr)
        for lambda_name, thread in threads_list:
            thread.join(timeout=30)  # 30 second timeout per thread
            if thread.is_alive():
                print(f"[TEARDOWN] WARNING: Thread {lambda_name} did not complete in time",
                      file=sys.stderr)

        # Check for exceptions from threads
        if not exceptions_queue.empty():
            print("\n" + "=" * 70, file=sys.stderr)
            print("THREAD EXCEPTIONS DETECTED DURING TEST", file=sys.stderr)
            print("=" * 70, file=sys.stderr)

            thread_exceptions = []
            while not exceptions_queue.empty():
                lambda_name, exc_info = exceptions_queue.get_nowait()
                thread_exceptions.append((lambda_name, exc_info))
                print(f"\nException in {lambda_name} thread:", file=sys.stderr)
                print(''.join(traceback.format_exception(*exc_info)), file=sys.stderr)

            print("=" * 70 + "\n", file=sys.stderr)

            # Re-raise the first exception to fail the test
            lambda_name, exc_info = thread_exceptions[0]
            raise exc_info[1].with_traceback(exc_info[2]) from None

        # Terminate moto server
        print("[TEARDOWN] Stopping moto server...", file=sys.stderr)
        moto_server.terminate()
        try:
            stdout, stderr = moto_server.communicate(timeout=5)
            if moto_server.returncode not in (0, -15):  # -15 is SIGTERM
                print(f"\n[TEARDOWN] Moto server had non-zero exit code: {moto_server.returncode}",
                      file=sys.stderr)
                if stdout:
                    print(f"STDOUT:\n{stdout}", file=sys.stderr)
                if stderr:
                    print(f"STDERR:\n{stderr}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print("[TEARDOWN] Moto server did not terminate, killing it...", file=sys.stderr)
            moto_server.kill()
            moto_server.wait()

        print("[TEARDOWN] Cleanup complete\n", file=sys.stderr)


def test_primary_tree_cover_loss(context):
    query = "select sum(area__ha) AS umd_tree_cover_loss__ha, sum(gfw_forest_carbon_gross_emissions__Mg_CO2e) AS gfw_forest_carbon_gross_emissions__Mg_CO2e from umd_tree_cover_loss__year where is__umd_regional_primary_forest_2001 = 'true' and (umd_tree_cover_density_2000__threshold >= 30 or is__umd_tree_cover_gain = 'true') group by umd_tree_cover_loss__year"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"
    assert result["data"]
    for row_actual, row_expected in zip(result["data"], IDN_24_9_PRIMARY_LOSS):
        assert row_actual["umd_tree_cover_loss__ha"] == pytest.approx(
            row_expected["area__ha"], 0.01
        )
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


def test_lat_lon_limit(context):
    query = "select latitude, longitude, umd_glad_landsat_alerts__date from umd_glad_landsat_alerts__date where umd_glad_landsat_alerts__date >= '2019-01-01' and umd_glad_landsat_alerts__date < '2020-01-01' order by umd_glad_landsat_alerts__date desc limit 100"
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

    lines = result["data"].splitlines()[1:]
    assert len(lines) == 100

    dates = [line.split(",")[2] for line in lines]
    assert dates[0] == "2019-12-28"

    # check the dates are already sorted in descending order
    assert list(sorted(dates, reverse=True)) == dates


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
    query = "select sum(area__ha) from umd_tree_cover_loss__year where umd_tree_cover_density_2000__threshold >= 30 group by umd_tree_cover_loss__year, tsc_tree_cover_loss_drivers__type order by umd_tree_cover_loss__year, tsc_tree_cover_loss_drivers__type"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"
    for row_actual, row_expected in zip(result["data"], IDN_24_9_LOSS_BY_DRIVER):
        assert row_actual["area__ha"] == pytest.approx(row_expected["area__ha"], 0.01)


def test_glad_alerts(context):
    query = "select count(*) from umd_glad_landsat_alerts__date where umd_glad_landsat_alerts__date >= '2019-01-01' and umd_glad_landsat_alerts__date < '2020-01-01' group by isoweek(umd_glad_landsat_alerts__date)"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"
    for row_actual, row_expected in zip(result["data"], IDN_24_9_GLAD_ALERTS):
        assert row_actual["count"] == row_expected["alert__count"]


def test_glad_alerts_count(context):
    query = "select count(*) from umd_glad_landsat_alerts__date where umd_glad_landsat_alerts__date >= '2019-01-01' and umd_glad_landsat_alerts__date < '2020-01-01' group by isoweek(umd_glad_landsat_alerts__date)"
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
        {"geometry": COD_21_4_GEOM, "query": query, "environment": DATA_ENVIRONMENT_10M},
        context,
    )

    assert result["status"] == "success"


def test_glad_s2_alerts(context):
    # TODO calculate number of alerts offline
    query = "select latitude, longitude, umd_glad_sentinel2_alerts__date, umd_glad_sentinel2_alerts__confidence from umd_glad_sentinel2_alerts__date where is__umd_regional_primary_forest_2001 = 'true' and umd_glad_sentinel2_alerts__date >= '2021-03-01'"
    result = tiled_handler(
        {"geometry": BRA_14_87_GEOM, "query": query, "environment": DATA_ENVIRONMENT_10M},
        context,
    )

    assert result["status"] == "success"


def test_land_cover_area(context):
    query = "select esa_land_cover_2015__class, sum(area__ha) AS esa_land_cover_2015__ha from esa_land_cover_2015__class group by esa_land_cover_2015__class"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"
    for row_actual, row_expected in zip(result["data"], IDN_24_9_ESA_LAND_COVER):
        assert (
            row_actual["esa_land_cover_2015__class"]
            == row_expected["esa_land_cover_2015__class"]
        )
        assert row_actual["esa_land_cover_2015__ha"] == pytest.approx(
            row_expected["area__ha"], 0.001
        )


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


def test_result_tiles_exceed_dynamodb_get_batch_limit(context, monkeypatch):
    monkeypatch.setattr(
        raster_analysis.results_store, "DYNAMODB_REQUEST_ITEMS_LIMIT", 2
    )
    query = "select count(umd_glad_landsat_alerts) from umd_glad_landsat_alerts__date where umd_glad_landsat_alerts__date >= '2019-01-01' and umd_glad_landsat_alerts__date < '2020-01-01' group by isoweek(umd_glad_landsat_alerts__date)"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"
    for row_actual, row_expected in zip(result["data"], IDN_24_9_GLAD_ALERTS):
        assert row_actual["count"] == row_expected["alert__count"]


def test_result_tiles_exceed_dynamodb_write_batch_limit(context, monkeypatch):
    monkeypatch.setattr(raster_analysis.results_store, "DYNAMODB_WRITE_ITEMS_LIMIT", 2)
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


def test_area_layers(context):
    query = "select sum(umd_tree_cover_loss__ha), sum(umd_tree_cover_loss_from_fires__ha) from data where umd_tree_cover_density_2000__threshold >= 30 group by umd_tree_cover_loss__year"
    result = tiled_handler(
        {"geometry": COD_21_4_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    assert result["status"] == "success"
    assert result["data"]
    for row_actual, row_expected in zip(result["data"], COD_24_1_TCLF):
        assert row_actual["umd_tree_cover_loss__ha"] == pytest.approx(
            row_expected["umd_tree_cover_loss__ha"], 0.01
        )
        assert row_actual["umd_tree_cover_loss_from_fires__ha"] == pytest.approx(
            row_expected["umd_tree_cover_loss_from_fires__ha"], 0.01
        )


def test_nonzero_no_data(context):
    """Test that nonzero NoData value for raster is respected. This means: 1)
    the NoData value isn't included in calculations 2) zero is included in
    calculations.

    Using the wri_tropical_tree_cover__percent raster, which has a
    NoData value of 255.
    """

    query = "select sum(area__ha) from wri_tropical_tree_cover__percent where wri_tropical_tree_cover__percent >= 0"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    expected_ttc_ha = 1291143.4962

    # 1291143.4962
    assert result["status"] == "success"
    assert result["data"][0]["area__ha"] == pytest.approx(expected_ttc_ha, 0.001)


def test_nonzero_no_data_group_by(context):
    """Test that nonzero NoData values are respected in GROUP BY operations.
    This means: 1) the NoData value will not be completed as a group value 2)
    zero can be included as a group value.

    Using the wri_tropical_tree_cover__percent raster, which has a
    NoData value of 255.
    """
    query = "select sum(area__ha) from data group by wri_tropical_tree_cover__percent"
    result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": query, "environment": DATA_ENVIRONMENT},
        context,
    )

    no_data_value = 255
    assert result["status"] == "success"
    assert any([row["wri_tropical_tree_cover__percent"] == 0 for row in result["data"]])
    assert all(
        [
            row["wri_tropical_tree_cover__percent"] != no_data_value
            for row in result["data"]
        ]
    )


def test_null_no_data(context):
    """Test that null NoData values are treated correctly."""

    fake_data_env = deepcopy(DATA_ENVIRONMENT)
    fake_data_env = [
        layer
        for layer in fake_data_env
        if layer["name"] == "wri_tropical_tree_cover__percent"
    ]
    fake_data_env[0]["no_data"] = None

    group_by_query = (
        "select sum(area__ha) from data group by wri_tropical_tree_cover__percent"
    )
    group_result = tiled_handler(
        {
            "geometry": IDN_24_9_GEOM,
            "query": group_by_query,
            "environment": fake_data_env,
        },
        context,
    )

    assert group_result["status"] == "success"
    assert any(
        [row["wri_tropical_tree_cover__percent"] == 0 for row in group_result["data"]]
    )
    assert any(
        [row["wri_tropical_tree_cover__percent"] == 255 for row in group_result["data"]]
    )

    base_query = "select sum(area__ha) from wri_tropical_tree_cover__percent"
    base_result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": base_query, "environment": fake_data_env},
        context,
    )
    assert base_result["status"] == "success"

    agg_query = "select sum(wri_tropical_tree_cover__percent) from data"
    agg_result = tiled_handler(
        {"geometry": IDN_24_9_GEOM, "query": agg_query, "environment": fake_data_env},
        context,
    )
    assert agg_result["status"] == "success"
