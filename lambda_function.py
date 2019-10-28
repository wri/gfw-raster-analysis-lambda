import logging
import sys

from shapely.geometry import shape

from raster_analysis import geoprocessing
from raster_analysis.geoprocessing import Filter
from raster_analysis.schemas import SCHEMA

# TODO this causes issues on AWS currently
# from lambda_decorators import, json_schema_validator

fmt = "%(asctime)s %(levelname)-4s - %(name)s - %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)

# See above TODO
# @json_schema_validator(request_schema=SCHEMA)
# @json_http_resp


def lambda_handler(event, context):
    logger.info("Test")
    analysis_raster_id = event["analysis_raster_id"]
    contextual_raster_ids = (
        event["contextual_raster_ids"] if "contextual_raster_ids" in event else []
    )
    aggregate_raster_ids = (
        event["aggregate_raster_ids"] if "aggregate_raster_ids" in event else []
    )
    density_raster_ids = (
        event["density_raster_ids"] if "density_raster_ids" in event else []
    )
    analyses = event["analyses"] if "analyses" in event else ["count", "area"]
    geometry = shape(event["geometry"])

    filter_raster_id = (
        event["filter_raster_id"] if "filter_raster_id" in event else None
    )
    filter_intervals = (
        event["filter_intervals"] if "filter_intervals" in event else None
    )

    get_area_summary = (
        event["get_area_summary"] if "get_area_summary" in event else False
    )

    return geoprocessing.analysis(
        geometry,
        analysis_raster_id,
        contextual_raster_ids,
        aggregate_raster_ids,
        filter_raster_id,
        filter_intervals,
        density_raster_ids,
        analyses,
        get_area_summary,
    )


if __name__ == "__main__":
    # "{\"analysis_raster_id\":\"loss\", \"contextual_raster_ids\":[\"wdpa\"], \"aggregate_raster_ids\":[\"tcd_2000\", \"tcd_2010\"], \"analysis\":\"sum\", \"geometry\":{\"type\":\"Polygon\",\"coordinates\":[[[9.0,4.1],[9.1,4.1],[9.1,4.2],[9.0,4.2],[9.0,4.1]]]},\"filters\":[{\"raster_id\":\"tcd_2000\",\"threshold\":30}]}"), None))

    print(
        lambda_handler(
            {
                "analysis_raster_id": "loss",
                # "contextual_raster_ids": ["ifl", "plantations"],
                "analyses": ["area", "sum"],
                "aggregate_raster_ids": ["biomass"],
                "density_raster_ids": ["biomass"],
                "get_area_summary": True,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-55.380185333356344, -12.276001055451573],
                            [-55.584550346447266, -12.613440960787752],
                            [-55.522765575047686, -13.145740529768766],
                            [-55.052250777466256, -13.293073446183152],
                            [-54.595994004053956, -13.007912962800466],
                            [-54.733821571022254, -12.242732332390261],
                            [-55.380185333356344, -12.276001055451573],
                        ]
                    ],
                },
                "filters": [{"raster_id": "tcd_2000", "threshold": 30}],
            },
            None,
        )
    )
    """

    print(
        lambda_handler(
            {
                "analysis_raster_id": "umd_landsat_alerts",
                "analyses": ["count"],
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                        [
                          -55.380185333356344,
                          -12.276001055451573
                        ],
                        [
                          -55.584550346447266,
                          -12.613440960787752
                        ],
                        [
                          -55.522765575047686,
                          -13.145740529768766
                        ],
                        [
                          -55.052250777466256,
                          -13.293073446183152
                        ],
                        [
                          -54.595994004053956,
                          -13.007912962800466
                        ],
                        [
                          -54.733821571022254,
                          -12.242732332390261
                        ],
                        [
                          -55.380185333356344,
                          -12.276001055451573
                        ]
                      ]
                    ]
                  },
                #"filters": [{"raster_id": "umd_landsat_alerts", "threshold": 20000}],
            },
            None,
        )
    )
    """
