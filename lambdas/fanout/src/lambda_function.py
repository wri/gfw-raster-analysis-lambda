from copy import deepcopy

from codeguru_profiler_agent import with_lambda_profiler

from raster_analysis.boto import invoke_lambda, lambda_client
from raster_analysis.globals import LOGGER, RASTER_ANALYSIS_LAMBDA_NAME


@with_lambda_profiler(profiling_group_name="raster_analysis_fanout_profiler")
def handler(event, context):
    tiles = event.get("tiles", [])
    payload_base = event["payload"]

    for tile in tiles:
        payload = deepcopy(payload_base)
        payload["tile"] = tile[1]
        payload["cache_id"] = tile[0]

        try:
            invoke_lambda(payload, RASTER_ANALYSIS_LAMBDA_NAME, lambda_client())
        except Exception as e:
            LOGGER.error(
                f"Invoke raster analysis lambda failed for aws request id: {context.aws_request_id}, tile: {tile}"
            )
            raise e
