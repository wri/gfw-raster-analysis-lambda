import csv
import os

import pytest
import requests

from canaries.results import (
    IDN_24_9_AREA,
    IDN_24_9_GAIN,
    IDN_24_9_GLAD_ALERTS,
    IDN_24_9_GLAD_ALERTS_2020_12,
    IDN_24_9_TCD_EXTENT_2000,
    IDN_24_9_TCL,
)

API_KEY = os.environ["API_KEY"]
SERVICE_URI = os.environ["SERVICE_URI"]

HEADERS = {"x-api-key": API_KEY, "origin": "globalforestwatch.org"}

IDN_24_9_GEOSTORE_ID = "c3833748f6815d31bad47d47f147c0f0"
ZONAL_URI = f"https://{SERVICE_URI}/analysis/zonal/{IDN_24_9_GEOSTORE_ID}"


def assert_zonal_tcl():
    params = {
        "sum": "area__ha",
        "group_by": "umd_tree_cover_loss__year",
        "filters": "umd_tree_cover_density_2000__30",
        "start_date": "2001-01-01",
        "end_date": "2020-01-01",
        "geostore_origin": "rw",
    }

    results = make_request(ZONAL_URI, params)
    for actual, expected in zip(results, IDN_24_9_TCL):
        assert actual["area__ha"] == pytest.approx(expected["area__ha"], 0.001)


def assert_zonal_tcd():
    params = {
        "sum": "area__ha",
        "filters": "umd_tree_cover_density_2000__30",
        "geostore_origin": "rw",
    }

    results = make_request(ZONAL_URI, params)
    assert results[0]["area__ha"] == pytest.approx(IDN_24_9_TCD_EXTENT_2000, 0.001)


def assert_zonal_area():
    params = {
        "sum": "area__ha",
        "geostore_origin": "rw",
    }

    results = make_request(ZONAL_URI, params)
    assert results[0]["area__ha"] == pytest.approx(IDN_24_9_AREA, 0.001)


def assert_zonal_gain():
    params = {
        "sum": "area__ha",
        "filters": "is__umd_tree_cover_gain",
        "geostore_origin": "rw",
    }

    results = make_request(ZONAL_URI, params)
    assert results[0]["area__ha"] == pytest.approx(IDN_24_9_GAIN, 0.001)


def assert_zonal_glad():
    params = {
        "sum": "alert__count",
        "group_by": "umd_glad_alerts__isoweek",
        "start_date": "2001-01-01",
        "end_date": "2021-01-01",
        "geostore_origin": "rw",
    }

    results = make_request(ZONAL_URI, params)
    for actual, expected in zip(results, IDN_24_9_GLAD_ALERTS):
        assert actual["count"] == expected["alert__count"]


def assert_query_tcl():
    params = {
        "sql": "select sum(area__ha) from umd_tree_cover_loss where umd_tree_cover_density_2000__threshold >= 30 group by umd_tree_cover_loss__year",
        "geostore_id": IDN_24_9_GEOSTORE_ID,
        "geostore_origin": "rw",
    }
    uri = f"https://{SERVICE_URI}/dataset/umd_tree_cover_loss/latest/query"

    results = make_request(uri, params)
    for actual, expected in zip(results, IDN_24_9_TCL):
        assert actual["area__ha"] == pytest.approx(expected["area__ha"], 0.001)


def assert_query_tcd():
    params = {
        "sql": "select sum(area__ha) from umd_tree_cover_density_2000 where umd_tree_cover_density_2000__threshold >= 30",
        "geostore_id": IDN_24_9_GEOSTORE_ID,
        "geostore_origin": "rw",
    }
    uri = f"https://{SERVICE_URI}/dataset/umd_tree_cover_density_2000/latest/query"

    results = make_request(uri, params)
    assert results[0]["area__ha"] == pytest.approx(IDN_24_9_TCD_EXTENT_2000, 0.001)


def assert_query_gain():
    params = {
        "sql": "select sum(area__ha) from umd_tree_cover_gain",
        "geostore_id": IDN_24_9_GEOSTORE_ID,
        "geostore_origin": "rw",
    }
    uri = f"https://{SERVICE_URI}/dataset/umd_tree_cover_gain/latest/query"

    results = make_request(uri, params)
    assert results[0]["area__ha"] == pytest.approx(IDN_24_9_GAIN, 0.001)


def assert_query_glad():
    params = {
        "sql": "select count(*) from umd_glad_landsat_alerts where umd_glad_landsat_alerts__date >= '2015-01-01' and umd_glad_landsat_alerts__date < '2021-01-01' group by isoweek(umd_glad_landsat_alerts__date)",
        "geostore_id": IDN_24_9_GEOSTORE_ID,
        "geostore_origin": "rw",
    }
    uri = f"https://{SERVICE_URI}/dataset/umd_glad_landsat_alerts/latest/query"

    results = make_request(uri, params)
    for actual, expected in zip(results, IDN_24_9_GLAD_ALERTS):
        assert actual["count"] == expected["alert__count"]


def assert_download_glad():
    params = {
        "sql": "select latitude, longitude, umd_glad_landsat_alerts__date, umd_glad_landsat_alerts__confidence from umd_glad_landsat_alerts where umd_glad_landsat_alerts__date >= '2020-12-01' and umd_glad_landsat_alerts__date < '2021-01-01'",
        "geostore_id": IDN_24_9_GEOSTORE_ID,
        "geostore_origin": "rw",
    }
    uri = f"https://{SERVICE_URI}/dataset/umd_glad_landsat_alerts/latest/download/csv"

    results = make_download_request(uri, params)
    assert len(results) == IDN_24_9_GLAD_ALERTS_2020_12 + 1

    results_csv = csv.DictReader(results)
    assert results_csv.fieldnames == [
        "latitude",
        "longitude",
        "umd_glad_landsat_alerts__date",
        "umd_glad_landsat_alerts__confidence",
    ]

    for row in results_csv:
        assert row["umd_glad_landsat_alerts__confidence"] in ["", "high"]
        assert "2020-12-01" <= row["umd_glad_landsat_alerts__date"] < "2021-01-01"
        assert 100 < float(row["longitude"]) < 105
        assert -1 < float(row["latitude"]) < 1


def make_request(uri, params):
    resp = requests.get(uri, params=params, headers=HEADERS)
    assert resp.status_code == 200 and resp.json()["status"] == "success"

    return resp.json()["data"]


def make_download_request(uri, params):
    resp = requests.get(uri, params=params, headers=HEADERS)
    assert resp.status_code == 200

    return resp.content.decode().splitlines()


def handler():
    # assert_zonal_tcl()
    # assert_zonal_tcd()
    assert_zonal_gain()
    assert_zonal_glad()
    assert_zonal_area()

    assert_query_tcl()
    assert_query_tcd()
    assert_query_gain()
    assert_query_glad()

    assert_download_glad()


handler()
