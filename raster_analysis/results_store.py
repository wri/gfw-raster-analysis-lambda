from decimal import Decimal
import os
from copy import deepcopy
from time import sleep
from typing import Dict, Any
from datetime import datetime, timedelta
from io import StringIO
from enum import Enum

import pandas as pd
from boto3.dynamodb.table import TableResource
from pandas import DataFrame

from raster_analysis.boto import dynamodb_resource, dynamodb_client
from raster_analysis.exceptions import RasterAnalysisException
from raster_analysis.globals import (
    RESULTS_CHECK_INTERVAL,
    RESULTS_CHECK_TRIES,
    DYMANODB_TTL_SECONDS,
    TILED_RESULTS_TABLE_NAME,
    TILED_STATUS_TABLE_NAME,
)


class ResultStatus(str, Enum):
    success = "success"
    error = "error"


class AnalysisResultsStore:
    def __init__(self, analysis_id: str):
        self._client: TableResource = dynamodb_resource().Table(TILED_RESULTS_TABLE_NAME)
        self.analysis_id: str = analysis_id

    def save_result(self, results: DataFrame, result_id: str) -> None:
        records_per_item = 5000
        curr_record = 0
        i = 0
        num_records = results.shape[0]
        items = []

        while curr_record < num_records:
            curr_df = results[curr_record:(curr_record + records_per_item)]

            csv_buf = StringIO()
            curr_df.to_csv(csv_buf, index=False, float_format="%.5f")

            item = {
                "PutRequest": {
                    "Item": {
                        "analysis_id": {"S": self.analysis_id},
                        "tile_id": {"S": f"{result_id}.{i}"},
                        "result": {"S": csv_buf.getvalue()},
                        "time_to_live": {"N": self._get_ttl()},
                    }
                }
            }
            items.append(item)

            curr_record += records_per_item
            i += 1

        dynamodb_client().batch_write_item(
            RequestItems={TILED_RESULTS_TABLE_NAME: items}
        )

        self.save_status(result_id, ResultStatus.success)

    def save_status(self, result_id: str, status: ResultStatus) -> None:
        dynamodb_client().put_item(
            TableName=TILED_STATUS_TABLE_NAME,
            Item={
                "analysis_id": {"S": self.analysis_id},
                "tile_id": {"S": result_id},
                "status": {"S": status.value},
                "time_to_live": {"N": self._get_ttl()},
            }
        )

    def get_results(self) -> Dict[str, Any]:
        results = dynamodb_client().query(
            ExpressionAttributeValues={":id": {"S": self.analysis_id}},
            KeyConditionExpression="analysis_id = :id",
            TableName=TILED_RESULTS_TABLE_NAME,
        )

        page_key = results.get("LastEvaluatedKey", None)
        while page_key:
            next_page = dynamodb_client().query(
                ExpressionAttributeValues={":id": {"S": self.analysis_id}},
                KeyConditionExpression="analysis_id = :id",
                TableName=TILED_RESULTS_TABLE_NAME,
                ExclusiveStartKey=page_key,
            )
            results["Items"] += next_page["Items"]
            page_key = next_page.get("LastEvaluatedKey", None)

        return results

    def get_statuses(self) -> Dict[str, Any]:
        return dynamodb_client().query(
            ExpressionAttributeValues={":id": {"S": self.analysis_id}},
            KeyConditionExpression="analysis_id = :id",
            TableName=TILED_STATUS_TABLE_NAME,
        )

    def wait_for_results(self, num_results: int) -> DataFrame:
        curr_count = 0
        tries = 0

        while curr_count < num_results and tries < RESULTS_CHECK_TRIES:
            sleep(RESULTS_CHECK_INTERVAL)
            tries += 1

            statuses_response = self.get_statuses()

            for item in statuses_response["Items"]:
                if item["status"]["S"] == ResultStatus.error:
                    raise RasterAnalysisException(
                        f"Tile {item['tile_id']} encountered an error, check logs."
                    )

            curr_count = statuses_response["Count"]

        if curr_count != num_results:
            raise TimeoutError(
                f"Timeout occurred before all lambdas completed. Result count: {num_results}; results completed: {curr_count}"
            )

        results_response = self.get_results()
        raw_results = [StringIO(item["result"]["S"]) for item in results_response["Items"]]
        dfs = [pd.read_csv(result) for result in raw_results]
        return pd.concat(dfs)

    @staticmethod
    def _get_ttl():
        return str(Decimal((datetime.now() + timedelta(seconds=DYMANODB_TTL_SECONDS)).timestamp()))
