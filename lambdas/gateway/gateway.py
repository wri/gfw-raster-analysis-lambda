import json
import boto3
import http.client
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GFW_API_URI = "production-api.globalforestwatch.org"
GLAD_UNCONFIRMED_CONST = 20000
GLAD_CONFIRMED_CONST = 30000


def lambda_handler(event, context):
    query_params = event["queryStringParameters"]
    multi_val_query_params = event["multiValueQueryStringParameters"]

    lambda_client = boto3.client("lambda")

    if "geometry_id" not in query_params:
        return {
            "isBase64Encoded": False,
            "statusCode": 400,
            "body": "Bad parameters: need geometry_id",
        }

    conn = http.client.HTTPConnection(GFW_API_URI)
    conn.request("GET", "/geostore/" + query_params["geometry_id"])
    response = conn.getresponse()

    if response.status == 404:
        return {
            "isBase64Encoded": False,
            "statusCode": 404,
            "body": "Geometry wih ID = "
            + query_params["geometry_id"]
            + " Not Found in GeoStore.",
        }

    body = json.loads(response.read())

    # Right now, just the first geometry
    geom = body["data"]["attributes"]["geojson"]["features"][0]["geometry"]
    areaHa = body["data"]["attributes"]["areaHa"]

    payload = dict()

    if "analysis_raster_id" in query_params:
        payload["analysis_raster_id"] = query_params["analysis_raster_id"]
    else:
        return {
            "isBase64Encoded": False,
            "statusCode": 400,
            "body": "Bad parameters: need analysis_raster_id",
        }

    if "analysis_type" in multi_val_query_params:
        payload["analyses"] = multi_val_query_params["analysis_type"]
    else:
        return {
            "isBase64Encoded": False,
            "statusCode": 400,
            "body": "Bad parameters: need analysis_type",
        }

    payload["geometry"] = geom

    if "contextual_raster_id" in multi_val_query_params:
        payload["contextual_raster_ids"] = multi_val_query_params[
            "contextual_raster_id"
        ]

    if "get_area_summary" in query_params:
        payload["get_area_summary"] = json.loads(query_params["get_area_summary"])

    if "filter_raster_id" in query_params:
        payload["filter_raster_id"] = query_params["filter_raster_id"]

        if "filter_threshold" in query_params:
            payload["filter_intervals"] = [[0, int(query_params["filter_threshold"])]]
        elif (
            "filter_date" in query_params
            and query_params["filter_raster_id"] == "glad_alerts"
        ):
            filter_date = datetime.strptime(
                query_params["filter_date"], "%Y-%m-%d"
            ).date()
            days_since_2015 = (filter_date - date(2015, 1, 1)).days
            payload["filter_intervals"] = [
                [20000, 20000 + days_since_2015],
                [30000, 30000 + days_since_2015],
            ]
        else:
            payload["filter_intervals"] = [[]]

    get_emissions = False
    if "aggregate_raster_id" in query_params:
        payload["aggregate_raster_ids"] = [query_params["aggregate_raster_id"]]

        if "emissions" in query_params["aggregate_raster_id"]:
            get_emissions = True
            payload["aggregate_raster_ids"] = ["biomass"]
            payload["density_raster_ids"] = ["biomass"]

    logger.info("Running raster analysis with params: " + json.dumps(payload))

    if areaHa < 5000000:
        lambda_response = lambda_client.invoke(
            FunctionName="raster-analysis",
            InvocationType="RequestResponse",
            Payload=bytes(json.dumps(payload), "utf-8"),
        )
    else:
        lambda_response = lambda_client.invoke(
            FunctionName="raster-analysis-tiled",
            InvocationType="RequestResponse",
            Payload=bytes(json.dumps(payload), "utf-8"),
        )

    response = json.loads(lambda_response["Payload"].read())

    if response["statusCode"] != 200:
        return {
            "isBase64Encoded": False,
            "statusCode": response["statusCode"],
            "headers": {"Content-Type": "application/json"},
            "body": response["body"],
        }

    body = json.loads(response["body"])

    # temp hook for emissions
    if get_emissions:
        body["detailed_table"]["emissions"] = body["detailed_table"]["biomass"]
        del body["detailed_table"]["biomass"]

    final_result = {"detailed_table": convert_to_csv_style(body["detailed_table"])}

    if "summary_table" in body:
        final_result["summary_table"] = convert_to_csv_style(body["summary_table"])

    return {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(final_result),
    }


def convert_to_csv_style(pandas_results):
    result_cols = list(pandas_results.keys())
    result_row_length = len(pandas_results[result_cols[0]])
    rows = []

    for i in range(0, result_row_length):
        row = dict()
        for col in result_cols:
            row[col] = pandas_results[col][str(i)]
            if col == "emissions":
                row[col] *= 0.5 * 44 / 12
        rows.append(row)

    return rows


if __name__ == "__main__":
    print(
        lambda_handler(
            {
                "queryStringParameters": {
                    "analysis_raster_id": "loss",
                    "geometry_id": "9e46feafae7e133b7fdf1a036eefeee6",
                    "filter_raster_id": "tcd_2000",
                    "filter_threshold": 30,
                    "aggregate_raster_id": "emissions",
                },
                "multiValueQueryStringParameters": {
                    "contextual_raster_id": ["wdpa"],
                    "analysis_type": ["count", "area"],
                },
            },
            None,
        )
    )
