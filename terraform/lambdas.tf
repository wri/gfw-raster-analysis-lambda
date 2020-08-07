resource "aws_lambda_function" "raster_analysis" {
  function_name    = substr("${local.project}-raster_analysis${local.name_suffix}", 0, 64)
  filename         = data.archive_file.lambda_fanout.output_path
  source_code_hash = data.archive_file.lambda_fanout.output_base64sha256
  role             = aws_iam_role.raster_analysis_lambda.arn
  runtime          = var,
  handler          = "lambda_function.handler"
  memory_size      = var.lambda_raster_analysis_memory_size
  timeout          = var.lambda_raster_analysis_timeout
  publish          = true
  tags             = local.tags
  layers           = [
    module.lambda_layers.raster_analysis_arn,
    data.terraform_remote_state.lambda-layers.outputs.lambda_layer_rasterio_arn
  ]

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      ENV                         = var.environment
      S3_BUCKET_DATA_LAKE         = data.terraform_remote_state.core.outputs.data-lake_bucket
      TILED_RESULTS_TABLE_NAME    = aws_dynamodb_table.tiled_results_table.name
    }
  }
}

resource "aws_lambda_function" "tiled_raster_analysis" {
  function_name    = substr("${local.project}-tiled_analysis${local.name_suffix}", 0, 64)
  filename         = data.archive_file.lambda_tiled_analysis.output_path
  source_code_hash = data.archive_file.lambda_tiled_analysis.output_base64sha256
  role             = aws_iam_role.raster_analysis_lambda.arn
  runtime          = var.lambda_tiled_analysis_runtime
  handler          = "lambda_function.handler"
  memory_size      = var.lambda_tiled_analysis_memory_size
  timeout          = var.lambda_tiled_analysis_timeout
  publish          = true
  tags             = local.tags
  layers           = [
    module.lambda_layers.raster_analysis_arn,
    data.terraform_remote_state.lambda-layers.outputs.lambda_layer_shapely_pyyaml_arn,
    data.terraform_remote_state.lambda-layers.outputs.lambda_layer_pandas_arn
  ]

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      ENV                         = var.environment
      S3_BUCKET_DATA_LAKE         = data.terraform_remote_state.core.outputs.data-lake_bucket
      RASTER_ANALYSIS_LAMBDA_NAME = aws_lambda_function.raster_analysis.function_name
      TILED_RESULTS_TABLE_NAME    = aws_dynamodb_table.tiled_results_table.name
    }
  }
}

resource "aws_lambda_function" "raster_analysis_fanout" {
  function_name    = substr("${local.project}-tiled_analysis${local.name_suffix}", 0, 64)
  filename         = data.archive_file.lambda_tiled_analysis.output_path
  source_code_hash = data.archive_file.lambda_tiled_analysis.output_base64sha256
  role             = aws_iam_role.raster_analysis_lambda.arn
  runtime          = var.lambda_tiled_analysis_runtime
  handler          = "lambda_function.handler"
  memory_size      = var.lambda_tiled_analysis_memory_size
  timeout          = var.lambda_tiled_analysis_timeout
  publish          = true
  tags             = local.tags
  layers           = [
    module.lambda_layers.raster_analysis_arn,
  ]

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      ENV                         = var.environment
      RASTER_ANALYSIS_LAMBDA_NAME = aws_lambda_function.raster_analysis.function_name
    }
  }
}