import os
from typing import Any, Dict, Optional
from uuid import uuid4, UUID

import geopandas as gpd
import pandas as pd
import tempfile
from aws_xray_sdk.core import patch, xray_recorder
from shapely.wkb import dumps as wkb_dumps

from raster_analysis.boto import s3_client
from raster_analysis.exceptions import QueryParseException
from raster_analysis.globals import LOGGER

patch(["boto3"])

# FIXME: Get these from env
BUCKET = "gfw-pipelines-test"
REGION = "us-east-1"


@xray_recorder.capture("Preprocessing")
def handler(event, context):
    try:
        LOGGER.info(f"Running preprocessing with parameters: {event}")
        fc: Optional[Dict] = event.get("feature_collection")
        uri: Optional[str] = event.get("uri")
        id_field = event["id_field"]  # How to make use of this?

        if fc is not None:
            gpdf = gpd.GeoDataFrame.from_features(fc, columns=[id_field])
        elif uri is not None:
            gpdf = gpd.read_file(uri, columns=[id_field])
        else:
            raise Exception("No valid input methods passed!")

        print("Here come the rows!")
        rows = []
        for record in gpdf.itertuples():
            geom_wkb = wkb_dumps(getattr(record, "geometry"))
            rows.append([getattr(record, id_field), geom_wkb])

        print(rows)

        # FIXME: Hash those args for cacheability!
        request_hash: UUID = uuid4()
        geom_prefix = f"test/otf_lists/{str(request_hash)}/geometries.csv"
        output_prefix = f"test/otf_lists/{str(request_hash)}/output"

        with tempfile.TemporaryDirectory() as tmp_dir:
            some_path = os.path.join(tmp_dir, "geometries.csv")
            df = pd.DataFrame(rows, columns=[id_field, 'geometry'])
            df.to_csv(some_path, index=False)

            upload_to_s3(some_path, BUCKET, geom_prefix)

        return {
            "status": "success",
            "geometries": {
                "bucket": BUCKET,
                "key": geom_prefix
            },
            "output": {
                "bucket": BUCKET,
                "prefix": output_prefix
            }
        }
    except QueryParseException as e:
        return {"status": "failed", "message": str(e)}
    except Exception as e:
        LOGGER.exception(e)
        return {"status": "error", "message": str(e)}


def upload_to_s3(path: str, bucket: str, dst: str) -> Dict[str, Any]:
    return s3_client().upload_file(path, bucket, dst)
