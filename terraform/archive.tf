data "archive_file" "lambda_raster_analysis" {
  type        = "zip"
  source_dir  = "../lambdas/raster_analysis/src"
  output_path = "../lambdas/raster_analysis/lambda.zip"
}

data "archive_file" "lambda_tiled_analysis" {
  type        = "zip"
  source_dir  = "../lambdas/tiled_analysis/src"
  output_path = "../lambdas/tiled_analysis/lambda.zip"
}

data "archive_file" "lambda_fan_out" {
  type        = "zip"
  source_dir  = "../lambdas/tiled_analysis/src"
  output_path = "../lambdas/tiled_analysis/lambda.zip"
}

data "archive_file" "lambda_raster_analysis_gateway" {
  type        = "zip"
  source_dir  = "../lambdas/raster_analysis_gateway/src"
  output_path = "../lambdas/raster_analysis_gateway/lambda.zip"
}