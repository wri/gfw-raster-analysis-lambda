import logging
import sys
import traceback

from shapely.geometry import shape

from raster_analysis import geoprocessing

from lambda_decorators import json_http_resp
from aws_xray_sdk.core import xray_recorder

fmt = "%(asctime)s %(levelname)-4s - %(name)s - %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)


@json_http_resp
def handler(event, context):
    with xray_recorder.capture("Raster Analysis Lambda") as subsegment:
        subsegment.put_annotation("RequestID", context.aws_request_id)
        subsegment.put_annotation("LogStream", context.log_stream_name)
        subsegment.put_metadata("RequestParams", event)

        try:
            logger.info(f"Running analysis with parameters: {event}")

            analyses = event["analyses"] if "analyses" in event else ["count", "area"]
            geometry = shape(event["geometry"])
            analysis_raster_id = (
                event["analysis_raster_id"] if "analysis_raster_id" in event else []
            )
            contextual_raster_ids = (
                event["contextual_raster_ids"]
                if "contextual_raster_ids" in event
                else []
            )
            aggregate_raster_ids = (
                event["aggregate_raster_ids"] if "aggregate_raster_ids" in event else []
            )
            extent_year = event["extent_year"] if "extent_year" in event else None
            threshold = event["threshold"] if "threshold" in event else None
            start = event["start"] if "start" in event else None
            end = event["end"] if "end" in event else None

            return geoprocessing.analysis(
                geometry,
                analyses,
                analysis_raster_id,
                contextual_raster_ids,
                aggregate_raster_ids,
                start,
                end,
                extent_year,
                threshold,
            )
        except Exception:
            logging.error(traceback.format_exc())
            raise Exception(f"Internal Server Error <{context.aws_request_id}>")


if __name__ == "__main__":
    print(
        handler(
            {
                "analyses": ["count", "area"],
                "analysis_raster_id": "loss",
                "contextual_raster_ids": ["wdpa"],
                "aggregate_raster_ids": ["biomass"],
                "extent_year": 2000,
                "threshold": 30,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-52.17705785667322, -12.5],
                            [-52.5, -12.5],
                            [-52.5, -11.25],
                            [-52.332613412228774, -11.25],
                            [-52.17705785667322, -12.5],
                        ]
                    ],
                },
            },
            {"log_stream_name": "test_log_stream", "aws_request_id": "test_id"},
        )
    )


"""
repro weird issue:

[-55.0, -12.5],
[-56.25, -12.5],
[-56.25, -11.25],
[-55.0, -11.25],
[-55.0, -12.5],

"""
