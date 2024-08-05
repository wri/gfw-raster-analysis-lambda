import os
import tempfile
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

import geopandas as gpd
import pandas as pd
from aws_xray_sdk.core import patch, xray_recorder
from shapely.wkb import dumps as wkb_dumps

from raster_analysis.boto import s3_client
from raster_analysis.exceptions import QueryParseException
from raster_analysis.globals import LOGGER, S3_PIPELINE_BUCKET

patch(["boto3"])


@xray_recorder.capture("Preprocessing")
def handler(event, context):
    try:
        LOGGER.info(f"Running preprocessing with parameters: {event}")
        fc: Optional[Dict] = event.get("feature_collection")
        uri: Optional[str] = event.get("uri")
        id_field = event.get("id_field", "fid")  # Reasonable to use a default?

        if fc is not None and uri is not None:
            raise Exception("Please specify GeoJSON via (only) one parameter!")
        elif fc is not None:
            gpdf = gpd.GeoDataFrame.from_features(fc, columns=[id_field, "geometry"])
        elif uri is not None:
            gpdf = gpd.read_file(uri, columns=[id_field, "geometry"])
        else:
            raise Exception("Please specify GeoJSON via (only) one parameter!")

        rows = []
        for record in gpdf.itertuples():
            geom_wkb = wkb_dumps(getattr(record, "geometry"), hex=True)
            rows.append([getattr(record, id_field), geom_wkb])

        # Consider replacing UUID with hash of args for cacheability
        request_hash: UUID = uuid4()
        geom_prefix = f"analysis/jobs/{str(request_hash)}/geometries.csv"
        output_prefix = f"analysis/jobs/{str(request_hash)}/output"

        with tempfile.TemporaryDirectory() as tmp_dir:
            some_path = os.path.join(tmp_dir, "geometries.csv")
            df = pd.DataFrame(rows, columns=[id_field, "geometry"])
            df.to_csv(some_path, index=False)

            upload_to_s3(some_path, S3_PIPELINE_BUCKET, geom_prefix)

        return {
            "status": "success",
            "geometries": {"bucket": S3_PIPELINE_BUCKET, "key": geom_prefix},
            "output": {"bucket": S3_PIPELINE_BUCKET, "prefix": output_prefix},
        }
    except QueryParseException as e:
        return {"status": "failed", "message": str(e)}
    except Exception as e:
        LOGGER.exception(e)
        return {"status": "error", "message": str(e)}


def upload_to_s3(path: str, bucket: str, dst: str) -> Dict[str, Any]:
    return s3_client().upload_file(path, bucket, dst)
