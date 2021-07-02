resource "aws_lambda_function" "raster_analysis" {
  function_name    = substr("${local.project}-raster_analysis${local.name_suffix}", 0, 64)
  filename         = data.archive_file.lambda_raster_analysis.output_path
  source_code_hash = data.archive_file.lambda_raster_analysis.output_base64sha256
  role             = aws_iam_role.raster_analysis_lambda.arn
  runtime          = var.lambda_raster_analysis_runtime
  handler          = "lambda_function.handler"
  memory_size      = var.lambda_raster_analysis_memory_size
  timeout          = var.lambda_raster_analysis_timeout
  publish          = true
  tags             = local.tags
  layers           = [
    module.lambda_layers.raster_analysis_arn,
    data.terraform_remote_state.lambda-layers.outputs.py37_rasterio_115_arn,
    data.terraform_remote_state.lambda-layers.outputs.py37_shapely_164_arn,
    data.terraform_remote_state.lambda-layers.outputs.py37_pandas_110_arn
  ]

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      ENV                         = var.environment
      S3_BUCKET_DATA_LAKE         = data.terraform_remote_state.core.outputs.data-lake_bucket
      TILED_RESULTS_TABLE_NAME    = aws_dynamodb_table.tiled_results_table.name
      TILED_STATUS_TABLE_NAME     = aws_dynamodb_table.tiled_status_table.name
      SETUPTOOLS_USE_DISTUTILS    = "stdlib"
    }
  }
}

resource "aws_lambda_function" "tiled_raster_analysis" {
  function_name    = substr("${local.project}-tiled_raster_analysis${local.name_suffix}", 0, 64)
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
    data.terraform_remote_state.lambda-layers.outputs.py37_rasterio_115_arn,
    data.terraform_remote_state.lambda-layers.outputs.py37_shapely_164_arn,
    data.terraform_remote_state.lambda-layers.outputs.py37_pandas_110_arn
  ]

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      ENV                         = var.environment
      S3_BUCKET_DATA_LAKE         = data.terraform_remote_state.core.outputs.data-lake_bucket
      RASTER_ANALYSIS_LAMBDA_NAME = aws_lambda_function.raster_analysis.function_name
      FANOUT_LAMBDA_NAME          = aws_lambda_function.raster_analysis_fanout.function_name
      TILED_RESULTS_TABLE_NAME    = aws_dynamodb_table.tiled_results_table.name
      TILED_STATUS_TABLE_NAME     = aws_dynamodb_table.tiled_status_table.name
      SETUPTOOLS_USE_DISTUTILS    = "stdlib"
    }
  }
}

resource "aws_lambda_function" "raster_analysis_fanout" {
  function_name    = substr("${local.project}-raster_analysis_fanout${local.name_suffix}", 0, 64)
  filename         = data.archive_file.lambda_fanout.output_path
  source_code_hash = data.archive_file.lambda_fanout.output_base64sha256
  role             = aws_iam_role.raster_analysis_lambda.arn
  runtime          = var.lambda_fanout_runtime
  handler          = "lambda_function.handler"
  memory_size      = var.lambda_fanout_memory_size
  timeout          = var.lambda_fanout_timeout
  publish          = true
  tags             = local.tags
  layers           = [
    module.lambda_layers.raster_analysis_arn,
    data.terraform_remote_state.lambda-layers.outputs.py37_shapely_164_arn,
  ]

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      ENV                         = var.environment
      RASTER_ANALYSIS_LAMBDA_NAME = aws_lambda_function.raster_analysis.function_name
      SETUPTOOLS_USE_DISTUTILS = "stdlib"
    }
  }
}