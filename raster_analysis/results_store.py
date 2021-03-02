from decimal import Decimal
import os
from copy import deepcopy
from time import sleep
from typing import List, Dict, Any
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
from boto3.dynamodb.table import TableResource

from raster_analysis.boto import dynamodb_resource
from raster_analysis.exceptions import RasterAnalysisException
from raster_analysis.globals import (
    RESULTS_CHECK_INTERVAL,
    RESULTS_CHECK_TRIES,
    DYMANODB_TTL_SECONDS,
)


class AnalysisResultsStore:
    def __init__(self, analysis_id: str):
        self._table_name: str = os.environ["TILED_RESULTS_TABLE_NAME"]
        self._client: TableResource = dynamodb_resource().Table(self._table_name)
        self.analysis_id: str = analysis_id

    def save_result(self, result: str, result_id: str) -> None:
        # store_result = self._convert_to_dynamo_format(result)
        self._client.put_item(
            Item={
                "analysis_id": self.analysis_id,
                "tile_id": result_id,
                "result": result,
                "time_to_live": self._get_ttl(),
                "error": False,
            }
        )

    def save_error(self, result_id: str) -> None:
        self._client.put_item(
            Item={
                "analysis_id": self.analysis_id,
                "tile_id": result_id,
                "result": [],
                "time_to_live": self._get_ttl(),
                "error": True,
            }
        )

    def get_results(self) -> Dict[str, Any]:
        return self._client.query(
            ExpressionAttributeValues={":id": self.analysis_id},
            KeyConditionExpression="analysis_id = :id",
            TableName=self._table_name,
        )

    def wait_for_results(self, num_results: int) -> List[Dict[str, Any]]:
        curr_count = 0
        tries = 0

        while curr_count < num_results and tries < RESULTS_CHECK_TRIES:
            sleep(RESULTS_CHECK_INTERVAL)
            tries += 1

            results_response = self.get_results()

            for item in results_response["Items"]:
                if item["error"]:
                    raise RasterAnalysisException(
                        f"Tile {item['tile_id']} encountered an error, check logs."
                    )

            curr_count = results_response["Count"]

        if curr_count != num_results:
            raise TimeoutError(
                f"Timeout occurred before all lambdas completed. Result count: {num_results}; results completed: {curr_count}"
            )

        raw_results = [StringIO(item["result"]) for item in results_response["Items"]]
        dfs = [pd.read_csv(result) for result in raw_results]
        return pd.concat(dfs)

    @staticmethod
    def _convert_from_dynamo_format(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        DynamoDB API returns Decimal objects for all numbers, so this is a util to
        convert the result back to ints and floats

        :param result: resulting dict from a call to raster analysis
        :return: result with int and float values instead of Decimal
        """
        converted_result = deepcopy(result)

        for layer, col in converted_result.items():
            if isinstance(col, list):
                if any(isinstance(val, Decimal) for val in col):
                    if all([val % 1 == 0 for val in col]):
                        converted_result[layer] = [int(val) for val in col]
                    else:
                        converted_result[layer] = [float(val) for val in col]
            else:
                if isinstance(col, Decimal):
                    converted_result[layer] = int(col) if col % 1 == 0 else float(col)

        return converted_result

    @staticmethod
    def _convert_to_dynamo_format(result: Dict[str, Any]) -> Dict[str, Any]:
        store_result = deepcopy(result)

        for layer, col in store_result.items():
            if isinstance(col, float):
                store_result[layer] = Decimal(str(col))
            elif isinstance(col, list) and len(col) > 0 and isinstance(col[0], float):
                store_result[layer] = [Decimal(str(val)) for val in col]

        return store_result

    @staticmethod
    def _get_ttl():
        return int(
            (datetime.now() + timedelta(seconds=DYMANODB_TTL_SECONDS)).timestamp()
        )
