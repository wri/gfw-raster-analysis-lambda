import os
import tempfile
from typing import Any, Dict, Optional, List
from uuid import UUID, uuid4

import geopandas as gpd
import pandas as pd
from aws_xray_sdk.core import patch, xray_recorder
from shapely.geometry import shape

from raster_analysis.boto import s3_client
from raster_analysis.globals import LOGGER, S3_PIPELINE_BUCKET
from raster_analysis.geometry import encode_geometry

import requests
from requests import Response
import json
from raster_analysis.boto import get_secrets_manager_client

patch(["boto3"])

GEOSTORE_PAGE_SIZE = 25


class UnexpectedResponseError(Exception):
    pass


@xray_recorder.capture("Preprocessing")
def handler(event: Dict[str, Any], context: Any) -> Any:
    try:
        LOGGER.info(f"Running preprocessing with parameters: {event}")
        fc: Optional[Dict] = event.get("feature_collection")
        uri: Optional[str] = event.get("uri")
        geostore_ids: Optional[List[str]] = event.get("geostore_ids")
        id_field: str = event.get("id_field", "fid")

        gpdf = None
        geostore_info = None
        if (fc and uri) or (fc and geostore_ids) or (uri and geostore_ids):
            raise Exception("Please specify exactly one of 'feature_collection', 'uri', or 'geostore_ids'.")
        elif fc is not None:
            gpdf = gpd.GeoDataFrame.from_features(fc, columns=[id_field, "geometry"])
        elif uri is not None:
            gpdf = gpd.read_file(uri, columns=[id_field, "geometry"])
        elif geostore_ids is not None:
            geostore_info = get_geostore_info(geostore_ids)
        else:
            raise Exception("Please specify exactly one of 'feature_collection', 'uri', or 'geostore_ids'.")

        if gpdf is not None and id_field not in gpdf.columns.tolist():
            raise Exception(f"Input feature collection is missing ID field '{id_field}'")

        rows: List[List[str]] = []
        if geostore_info is not None:
            for info in geostore_info:
                # Use the geostoreId itself as the id field for the output.
                id = info["geostoreId"]
                # The RW find-by-ids call returns the geometry as a feature collection,
                # which I think should always have one feature (?)
                fc = info["geostore"]["data"]["attributes"]["geojson"]["features"]
                if fc is None:
                    raise Exception(f"Missing features attribute for geostore '{id}'")
                # GeoDataFrame.from_features() expects each feature to have a
                # 'properties' field.
                for f in fc:
                    if f.get("properties") is None:
                        f["properties"] = {}
                minidf = gpd.GeoDataFrame.from_features(fc)
                geom = shape(getattr(minidf.iloc[0], "geometry"))
                encoded_geom = encode_geometry(geom)
                rows.append([id, encoded_geom])
        else:
            assert(gpdf is not None)
            for record in gpdf.itertuples():
                geom = shape(getattr(record, "geometry"))
                encoded_geom = encode_geometry(geom)
                rows.append([getattr(record, id_field), encoded_geom])

        # Consider replacing UUID with hash of args for cacheability
        request_hash: UUID = uuid4()
        geom_prefix = f"analysis/jobs/{str(request_hash)}/geometries.csv"
        output_prefix = f"analysis/jobs/{str(request_hash)}/output"

        with tempfile.TemporaryDirectory() as tmp_dir:
            some_path = os.path.join(tmp_dir, "geometries.csv")
            # In the file sent to the distributed map, use the standard name 'fid'
            # for the id field, to make the step function code simpler.
            df = pd.DataFrame(rows, columns=["fid", "geometry"])
            df.to_csv(some_path, index=False)

            upload_to_s3(some_path, S3_PIPELINE_BUCKET, geom_prefix)

        return {
            "status": "success",
            "geometries": {"bucket": S3_PIPELINE_BUCKET, "key": geom_prefix},
            "output": {"bucket": S3_PIPELINE_BUCKET, "prefix": output_prefix},
        }
    except Exception as e:
        LOGGER.exception(e)
        return {"status": "error", "message": str(e)}


def upload_to_s3(path: str, bucket: str, dst: str) -> Dict[str, Any]:
    return s3_client().upload_file(path, bucket, dst)


# Similar to gfw-datapump:src/datapump/sync/rw_areas.py:get_geostore
def get_geostore_info(geostore_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get a list of Geostore information (including geometries) from a list of
    geostore IDs
    """

    LOGGER.info("Get geostore info by IDs")

    headers: Dict[str, str] = {
        "Content-Type": "application/json"
        # Should we supply a bearer token or an API key, even though neither is required?
        # "Authorization": f"Bearer {token()}"
    }
    url: str = (
        "https://api.resourcewatch.org/v2/geostore/find-by-ids"
    )
    geostores: List[Dict[str, Any]] = []

    for i in range(0, len(geostore_ids), GEOSTORE_PAGE_SIZE):
        payload: Dict[str, List[str]] = {
            "geostores": geostore_ids[i: i + GEOSTORE_PAGE_SIZE]
        }

        retries = 0
        while retries < 2:
            r: Response = requests.post(url, json=payload, headers=headers)

            if r.status_code != 200:
                retries += 1
                if retries > 1:
                    raise UnexpectedResponseError(
                        f"geostore/find-by-ids returned response {r.status_code} on block {i}"
                    )
            else:
                geostores += r.json()["data"]
                break

    return geostores


TOKEN = None


def token() -> str:
    global TOKEN
    if TOKEN is None:
        TOKEN = _get_token()
    return TOKEN


def _get_token() -> str:
    response = get_secrets_manager_client().get_secret_value(
        SecretId="gfw-api/token"
    )
    return json.loads(response["SecretString"])["token"]
