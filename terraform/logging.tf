resource "aws_cloudwatch_log_group" "raster_analysis" {
  name              = "/aws/lambda/${aws_lambda_function.raster_analysis.function_name}"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "tiled_raster_analysis" {
  name              = "/aws/lambda/${aws_lambda_function.tiled_raster_analysis.function_name}"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "raster_analysis_fanout" {
  name              = "/aws/lambda/${aws_lambda_function.raster_analysis_fanout.function_name}"
  retention_in_days = 30
}