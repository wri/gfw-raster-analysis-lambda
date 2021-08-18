from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from hashlib import md5
from io import StringIO
from time import sleep
from typing import Any, Dict, List

import pandas as pd
from boto3.dynamodb.table import TableResource
from pandas import DataFrame
from shapely.geometry import Polygon

from raster_analysis.boto import dynamodb_client, dynamodb_resource
from raster_analysis.exceptions import RasterAnalysisException
from raster_analysis.globals import (
    RESULTS_CACHE_TTL_SECONDS,
    RESULTS_CHECK_INTERVAL,
    RESULTS_CHECK_TRIES,
    TILED_RESULTS_TABLE_NAME,
    TILED_STATUS_TABLE_NAME,
    BasePolygon,
)


class ResultStatus(str, Enum):
    success = "success"
    error = "error"


class AnalysisResultsStore:
    def __init__(self):
        self._client: TableResource = dynamodb_resource().Table(
            TILED_RESULTS_TABLE_NAME
        )

    def save_result(self, results: DataFrame, result_id: str) -> None:
        records_per_item = 5000
        curr_record = 0
        i = 0
        num_records = results.shape[0]
        items = []

        while curr_record < num_records:
            curr_df = results[curr_record : (curr_record + records_per_item)]

            csv_buf = StringIO()
            curr_df.to_csv(csv_buf, index=False, float_format="%.5f")

            item = {
                "PutRequest": {
                    "Item": {
                        "tile_id": {"S": f"{result_id}"},
                        "part_id": {"N": f"{i}"},
                        "result": {"S": csv_buf.getvalue()},
                        "time_to_live": {"N": self._get_ttl()},
                    }
                }
            }
            items.append(item)

            curr_record += records_per_item
            i += 1

        if items:
            dynamodb_client().batch_write_item(
                RequestItems={TILED_RESULTS_TABLE_NAME: items}
            )

        self.save_status(result_id, ResultStatus.success, i + 1)

    def save_status(
        self,
        result_id: str,
        status: ResultStatus,
        parts: int,
        detail: str = " ",
    ) -> None:
        dynamodb_client().put_item(
            TableName=TILED_STATUS_TABLE_NAME,
            Item={
                "tile_id": {"S": result_id},
                "status": {"S": status.value},
                "detail": {"S": detail},
                "parts": {"N": str(parts)},
                "time_to_live": {"N": self._get_ttl()},
            },
        )

    def get_results(self, tiles: List) -> List[Dict[str, Any]]:

        # Fetching result parts for tile from status table to able to batch_get
        # results that requires secondary index
        result_statuses = self.get_statuses(tiles)
        if len(result_statuses) == 0:
            return []

        tiles_and_result_parts = []
        for status in result_statuses:
            for part_id in range(int(status["parts"]["N"])):
                tiles_and_result_parts.append(
                    {
                        "tile_id": {"S": status["tile_id"]["S"]},
                        "part_id": {"N": str(part_id)},
                    }
                )

        results = self._get_batch_items(
            TILED_RESULTS_TABLE_NAME, tiles_and_result_parts
        )

        return results

    def get_statuses(
        self, tile_ids=List[str], status_filter: ResultStatus = None
    ) -> List[Dict[str, Any]]:
        batch_tiles = [{"tile_id": {"S": tile_id}} for tile_id in tile_ids]
        statuses = self._get_batch_items(TILED_STATUS_TABLE_NAME, batch_tiles)

        if status_filter:
            statuses = [
                status
                for status in statuses
                if status["status"]["S"] == status_filter.success
            ]

        return statuses

    def wait_for_results(
        self, lambda_tiles: List[str], all_tiles: List[str]
    ) -> DataFrame:
        curr_count = 0
        tries = 0
        num_results = len(lambda_tiles)

        while curr_count < len(lambda_tiles) and tries < RESULTS_CHECK_TRIES:
            sleep(RESULTS_CHECK_INTERVAL)
            tries += 1

            statuses = self.get_statuses(lambda_tiles)
            for item in statuses:
                if item["status"]["S"] == ResultStatus.error:
                    raise RasterAnalysisException(
                        f"Tile {item['tile_id']} encountered error: {item['detail']}"
                    )

            curr_count = len(statuses)

        if curr_count != num_results:
            raise TimeoutError(
                f"Timeout occurred before all lambdas completed. Result count: {num_results}; results completed: {curr_count}"
            )

        result_items = self.get_results(all_tiles)
        raw_results = [StringIO(item["result"]["S"]) for item in result_items]

        dfs = [pd.read_csv(result) for result in raw_results]
        results = pd.concat(dfs) if dfs else pd.DataFrame()
        print(f"Result dataframe: {results.to_dict()}")
        return results

    @staticmethod
    def get_cache_key(tile: Polygon, geom: BasePolygon, query: str) -> str:
        """Create md5 has for tile-geom_overlap-query result."""
        geom_tile_intersection = tile.intersection(geom)
        key = f"{query}-{tile.wkt}-{geom_tile_intersection.wkt}"

        return md5(key.encode()).hexdigest()

    @staticmethod
    def _get_ttl():
        return str(
            Decimal(
                (
                    datetime.now() + timedelta(seconds=int(RESULTS_CACHE_TTL_SECONDS))
                ).timestamp()
            )
        )

    @staticmethod
    def _get_batch_items(table_name, keys):
        results_response = dynamodb_client().batch_get_item(
            RequestItems={table_name: {"Keys": keys}}
        )
        results = results_response["Responses"][table_name]
        if len(results) == 0:
            return results

        unprocessed = results_response["UnprocessedKeys"]
        while len(unprocessed) > 0:
            results_response = dynamodb_client().batch_get_item(
                RequestItems={table_name: {"Keys": unprocessed[table_name]["Keys"]}}
            )
            results += results_response["Responses"][table_name]
            unprocessed = results_response["UnprocessedKeys"]

        return results
