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

data "archive_file" "lambda_fanout" {
  type        = "zip"
  source_dir  = "../lambdas/fanout/src"
  output_path = "../lambdas/fanout/lambda.zip"
}

data "archive_file" "lambda_preprocessing" {
  type        = "zip"
  source_dir  = "../lambdas/preprocessing/src"
  output_path = "../lambdas/preprocessing/lambda.zip"
}

data "archive_file" "lambda_list_tiled_analysis" {
  type        = "zip"
  source_dir  = "../lambdas/list_tiled_analysis/src"
  output_path = "../lambdas/list_tiled_analysis/lambda.zip"
}

data "archive_file" "lambda_aggregation" {
  type        = "zip"
  source_dir  = "../lambdas/aggregation/src"
  output_path = "../lambdas/aggregation/lambda.zip"
}
