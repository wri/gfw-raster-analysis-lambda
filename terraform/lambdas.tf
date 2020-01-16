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
    data.terraform_remote_state.core.outputs.lambda_layer_rasterio_arn
  ]

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      ENV                 = var.environment
      S3_BUCKET_DATA_LAKE = data.terraform_remote_state.core.outputs.data-lake_bucket
    }
  }
}

resource "aws_lambda_function" "tiled_analysis" {
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
    data.terraform_remote_state.core.outputs.lambda_layer_rasterio_arn,
  ]

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      ENV                         = var.environment
      S3_BUCKET_DATA_LAKE         = data.terraform_remote_state.core.outputs.data-lake_bucket
      RASTER_ANALYSIS_LAMBDA_NAME = aws_lambda_function.raster_analysis.function_name
    }
  }
}

resource "aws_lambda_function" "raster_analysis_gateway" {
  function_name    = substr("${local.project}-raster_analysis_gateway${local.name_suffix}", 0, 64)
  filename         = data.archive_file.lambda_raster_analysis_gateway.output_path
  source_code_hash = data.archive_file.lambda_raster_analysis_gateway.output_base64sha256
  role             = aws_iam_role.raster_analysis_lambda.arn
  runtime          = var.lambda_raster_analysis_gateway_runtime
  handler          = "lambda_function.handler"
  memory_size      = var.lambda_raster_analysis_gateway_memory_size
  timeout          = var.lambda_raster_analysis_gateway_timeout
  publish          = true
  tags             = local.tags
  layers           = [
    module.lambda_layers.raster_analysis_arn
  ]


  tracing_config {
    mode = "Active"
  }

  environment {
    variables = {
      ENV                         = var.environment
      S3_BUCKET_DATA_LAKE         = data.terraform_remote_state.core.outputs.data-lake_bucket
      RASTER_ANALYSIS_LAMBDA_NAME = aws_lambda_function.raster_analysis.function_name
      TILED_ANALYSIS_LAMBDA_NAME  = aws_lambda_function.tiled_analysis.function_name
    }
  }
}