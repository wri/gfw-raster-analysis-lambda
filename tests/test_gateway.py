import os
import json
from mock import patch

os.environ["RASTER_ANALYSIS_LAMBDA_NAME"] = "test_raster_analysis"
os.environ["TILED_ANALYSIS_LAMBDA_NAME"] = "test_tiled_analysis"
os.environ["ENV"] = "dev"

from lambdas.raster_analysis_gateway.src.lambda_function import (
    get_geostore,
    get_gladalerts_date,
    handler,
    get_raster_analysis_payload,
)


def test_get_geostore(requests_mock):
    requests_mock.get(
        "https://staging-api.globalforestwatch.org/geostore/test_geostore_id",
        json=TEST_GEOSTORE_RESPONSE,
    )

    geom, area = get_geostore("test_geostore_id")
    assert (
        geom
        == TEST_GEOSTORE_RESPONSE["data"]["attributes"]["geojson"]["features"][0][
            "geometry"
        ]
    )  # TODO: get all geometries?
    assert area == TEST_GEOSTORE_RESPONSE["data"]["attributes"]["areaHa"]


def test_get_gladalerts_date():
    date = "2020-01-01"
    days_since_2015 = 31826

    assert get_gladalerts_date(date) == days_since_2015


def test_get_raster_analysis_payload():
    payload = get_raster_analysis_payload(
        {"feature": "fake"},
        {
            "geostore_id": "test_geostore_id",
            "extent_year": "2000",
            "threshold": "30",
            "aggregate_raster_id": "emissions",
            "start": "2005",
            "end": "2018",
        },
        {"contextual_raster_id": ["wdpa", "ifl"]},
        "/analysis/treecoverloss",
    )

    assert payload["geometry"] == {"feature": "fake"}
    assert payload["analysis_raster_id"] == "loss"
    assert payload["threshold"] == 30
    assert payload["extent_year"] == 2000
    assert payload["aggregate_raster_id"] == "emissions"
    assert payload["contextual_raster_id"] == ["wdpa", "ifl"]
    assert payload["analyses"] == ["area"]
    assert payload["start"] == 5
    assert payload["end"] == 18

    payload = get_raster_analysis_payload(
        {"feature": "fake"},
        {
            "geostore_id": "test_geostore_id",
            "contextual_raster_id": "wdpa",
            "start": "2019-01-01",
            "end": "2019-12-10",
        },
        {},
        "/analysis/gladalerts",
    )

    assert payload["geometry"] == {"feature": "fake"}
    assert payload["analysis_raster_id"] == "glad_alerts"
    assert payload["contextual_raster_id"] == "wdpa"
    assert payload["analyses"] == ["count"]
    assert payload["start"] == 31461
    assert payload["end"] == 31804

    payload = get_raster_analysis_payload(
        {"feature": "fake"},
        {"geostore_id": "test_geostore_id", "contextual_raster_id": "wdpa"},
        {},
        "/analysis/summary",
    )

    assert "analysis_raster_id" not in payload
    assert payload["contextual_raster_id"] == "wdpa"
    assert payload["analyses"] == ["area"]


@patch("lambdas.raster_analysis_gateway.src.lambda_function.run_raster_analysis")
def test_handler(mock_run_analysis, requests_mock):
    requests_mock.get(
        "https://staging-api.globalforestwatch.org/geostore/test_geostore_id",
        json=TEST_GEOSTORE_RESPONSE,
    )

    mock_run_analysis.return_value = {
        "loss": [1, 1, 2, 2, 3, 3],
        "wdpa": [0, 1, 0, 1, 0, 1],
        "area": [4, 6, 1, 5, 2, 5],
    }

    result = handler(
        {
            "queryStringParameters": {
                "geostore_id": "test_geostore_id",
                "extent_year": "2000",
                "threshold": "30",
                "aggregate_raster_id": "emissions",
                "contextual_raster_id": "wdpa",
            },
            "multiValueQueryStringParameters": {
                "contextual_raster_id": ["wdpa"],
                "threshold": ["30"],
            },
            "path": "/analysis/treecoverloss",
        },
        None,
    )

    assert result["statusCode"] == 200
    assert json.loads(result["body"]) == [
        {"loss": 2001, "wdpa": 0, "area": 4},
        {"loss": 2001, "wdpa": 1, "area": 6},
        {"loss": 2002, "wdpa": 0, "area": 1},
        {"loss": 2002, "wdpa": 1, "area": 5},
        {"loss": 2003, "wdpa": 0, "area": 2},
        {"loss": 2003, "wdpa": 1, "area": 5},
    ]

    mock_run_analysis.return_value = {
        "gladalerts": [31001, 31001, 31002, 31002, 31003, 31003],
        "wdpa": [0, 1, 0, 1, 0, 1],
        "count": [4, 6, 1, 5, 2, 5],
    }

    result = handler(
        {
            "queryStringParameters": {
                "geostore_id": "test_geostore_id",
                "extent_year": "2000",
                "threshold": "30",
                "aggregate_raster_id": "emissions",
                "contextual_raster_id": "wdpa",
            },
            "multiValueQueryStringParameters": {
                "contextual_raster_id": ["wdpa"],
                "threshold": ["30"],
            },
            "path": "/analysis/gladalerts",
        },
        None,
    )

    assert result["statusCode"] == 200
    assert json.loads(result["body"]) == [
        {"gladalerts": "2017-09-28", "wdpa": 0, "count": 4},
        {"gladalerts": "2017-09-28", "wdpa": 1, "count": 6},
        {"gladalerts": "2017-09-29", "wdpa": 0, "count": 1},
        {"gladalerts": "2017-09-29", "wdpa": 1, "count": 5},
        {"gladalerts": "2017-09-30", "wdpa": 0, "count": 2},
        {"gladalerts": "2017-09-30", "wdpa": 1, "count": 5},
    ]


TEST_GEOSTORE_RESPONSE = {
    "data": {
        "type": "geoStore",
        "id": "test_geostore_id",  # pragma: allowlist secret
        "attributes": {
            "geojson": {
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [8.923_828_125_016_8, 9.556_965_798_610_79],
                                    [12.791_015_625_021_4, 8.689_186_410_236_46],
                                    [8.923_828_125_016_8, 4.849_697_804_134_93],
                                    [8.923_828_125_016_8, 9.556_965_798_610_79],
                                ]
                            ],
                        },
                    }
                ],
                "crs": {},
                "type": "FeatureCollection",
            },
            "hash": "069b603da1c881cf0fc193c39c3687bb",  # pragma: allowlist secret
            "provider": {},
            "areaHa": 11_186_986.783_767_128,
            "bbox": [
                8.923_828_125_016_8,
                4.849_697_804_134_93,
                12.791_015_625_021_4,
                9.556_965_798_610_79,
            ],
            "lock": False,
            "info": {"use": {}},
        },
    }
}
