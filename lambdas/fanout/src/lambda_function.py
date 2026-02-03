import os
from copy import deepcopy

from raster_analysis.boto import invoke_lambda, lambda_client
from raster_analysis.globals import LOGGER, RASTER_ANALYSIS_LAMBDA_NAME

try:
    if os.getenv("DISABLE_CODEGURU", "").lower() in ("1", "true", "yes"):
        raise ImportError()
    from codeguru_profiler_agent import with_lambda_profiler
except ImportError:
    def with_lambda_profiler(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


@with_lambda_profiler(profiling_group_name="raster_analysis_fanout_profiler")
def handler(event, context):
    tiles = event.get("tiles", [])
    payload_base = event["payload"]

    bench_invoker = event.get("__bench_invoker__")

    for tile in tiles:
        payload = deepcopy(payload_base)
        payload["tile"] = tile[1]
        payload["cache_id"] = tile[0]

        try:
            if bench_invoker is not None:
                bench_invoker.invoke(payload, RASTER_ANALYSIS_LAMBDA_NAME)
            else:
                invoke_lambda(payload, RASTER_ANALYSIS_LAMBDA_NAME, lambda_client())
        except Exception as e:
            LOGGER.error(
                f"Invoke raster analysis lambda failed for aws request id: {context.aws_request_id}, tile: {tile}"
            )
            raise e
