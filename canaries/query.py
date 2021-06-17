import os

import pytest
import requests

from canaries.results import (
    IDN_24_9_GAIN,
    IDN_24_9_GLAD_ALERTS,
    IDN_24_9_TCD_EXTENT_2000,
    IDN_24_9_TCL,
)

API_KEY = os.environ["API_KEY"]
SERVICE_URI = os.environ["SERVICE_URI"]

HEADERS = {"x-api-key": API_KEY, "origin": "globalforestwatch.org"}

IDN_24_9_GEOSTORE_ID = "c3833748f6815d31bad47d47f147c0f0"


def assert_tcl():
    params = {
        "sql": "select sum(area__ha) from umd_tree_cover_loss where umd_tree_cover_density_2000__threshold >= 30 group by umd_tree_cover_loss__year",
        "geostore_id": IDN_24_9_GEOSTORE_ID,
        "geostore_origin": "rw",
    }
    uri = f"https://{SERVICE_URI}/dataset/umd_tree_cover_loss/latest/query"

    results = make_query_request(uri, params)
    for actual, expected in zip(results, IDN_24_9_TCL):
        assert actual["area__ha"] == pytest.approx(expected["area__ha"], 0.001)


def assert_tcd():
    params = {
        "sql": "select sum(area__ha) from umd_tree_cover_density_2000 where umd_tree_cover_density_2000__threshold >= 30",
        "geostore_id": IDN_24_9_GEOSTORE_ID,
        "geostore_origin": "rw",
    }
    uri = f"https://{SERVICE_URI}/dataset/umd_tree_cover_density_2000/latest/query"

    results = make_query_request(uri, params)
    assert results[0]["area__ha"] == pytest.approx(IDN_24_9_TCD_EXTENT_2000, 0.001)


def assert_gain():
    params = {
        "sql": "select sum(area__ha) from umd_tree_cover_gain",
        "geostore_id": IDN_24_9_GEOSTORE_ID,
        "geostore_origin": "rw",
    }
    uri = f"https://{SERVICE_URI}/dataset/umd_tree_cover_gain/latest/query"

    results = make_query_request(uri, params)
    assert results[0]["area__ha"] == pytest.approx(IDN_24_9_GAIN, 0.001)


def assert_glad():
    params = {
        "sql": "select count(*) from umd_glad_landsat_alerts where umd_glad_landsat_alerts__date >= '2015-01-01' and umd_glad_landsat_alerts__date < '2021-01-01' group by isoweek(umd_glad_landsat_alerts__date)",
        "geostore_id": IDN_24_9_GEOSTORE_ID,
        "geostore_origin": "rw",
    }
    uri = f"https://{SERVICE_URI}/dataset/umd_glad_landsat_alerts/latest/query"

    results = make_query_request(uri, params)
    for actual, expected in zip(results, IDN_24_9_GLAD_ALERTS):
        assert actual["count"] == expected["alert__count"]


def make_query_request(uri, params):
    resp = requests.get(uri, params=params, headers=HEADERS)
    assert resp.status_code == 200 and resp.json()["status"] == "success"

    return resp.json()["data"]


assert_tcl()
assert_tcd()
assert_gain()
assert_glad()
