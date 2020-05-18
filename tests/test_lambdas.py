from mock import patch
import os
import json

os.environ["RASTER_ANALYSIS_LAMBDA_NAME"] = "test_raster_analysis"

from lambdas.tiled_analysis.src.lambda_function import handler


@patch("raster_analysis.tiling.run_raster_analysis")
def test_tiled_analysis_lambda(mock_run_raster_analysis):
    class Context(object):
        pass

    context = Context()
    context.aws_request_id = "test_tiled_id"
    context.log_stream_name = "test_log_stream"

    mock_run_raster_analysis.side_effect = [
        {
            "ras0": [1, 1, 2, 4, 4, 7, 7, 9],
            "ras1": [0, 1, 1, 0, 1, 0, 1, 0],
            "count": [3, 8, 4, 2, 9, 4, 5, 8],
            "ras2": [3.2, 8.2, 4.2, 2.2, 9.2, 4.2, 5.2, 8.2],
        },
        {
            "ras0": [1, 2, 2, 3, 3, 6, 7, 8, 9, 9, 10],
            "ras1": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            "count": [6, 2, 1, 4, 3, 2, 9, 8, 4, 1, 6],
            "ras2": [6.2, 2.2, 1.2, 4.2, 3.2, 2.2, 9.2, 8.2, 4.2, 1.2, 6.2],
        },
    ]

    response = handler(
        {
            "analysis_raster_id": "ras0",
            "contextual_raster_ids": ["ras1"],
            "aggregate_raster_ids": ["ras2"],
            "analysis": "count",
            "geometry": {
                "type": "Polygon",
                "coordinates": (
                    ((1.0, 0.2), (1.0, 2.4), (0.2, 2.4), (0.2, 0.2), (1.0, 0.2)),
                ),
            },
            "extent_year": 2000,
            "threshold": 30,
        },
        context,
    )

    result = json.loads(response["body"])
    assert result["ras0"] == [1, 1, 2, 2, 3, 3, 4, 4, 6, 7, 7, 8, 9, 9, 10]
    assert result["ras1"] == [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    assert result["count"] == [
        9.0,
        8.0,
        2.0,
        5.0,
        4.0,
        3.0,
        2.0,
        9.0,
        2.0,
        4.0,
        14.0,
        8.0,
        12.0,
        1.0,
        6.0,
    ]
    assert len(result["ras2"]) == 15
    assert json.dumps(result)
