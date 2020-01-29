resource "aws_api_gateway_rest_api" "raster_analysis_gateway" {
  name = substr("${local.project}-raster_analysis_gateway${local.name_suffix}", 0, 64)
}

resource "aws_api_gateway_resource" "analysis" {
  rest_api_id = aws_api_gateway_rest_api.raster_analysis_gateway.id
  parent_id   = aws_api_gateway_rest_api.raster_analysis_gateway.root_resource_id
  path_part   = "analysis"
}

resource "aws_api_gateway_resource" "treecoverloss" {
  rest_api_id = aws_api_gateway_rest_api.raster_analysis_gateway.id
  parent_id   = aws_api_gateway_resource.analysis.id
  path_part   = "treecoverloss"
}

resource "aws_api_gateway_resource" "gladalerts" {
  rest_api_id = aws_api_gateway_rest_api.raster_analysis_gateway.id
  parent_id   = aws_api_gateway_resource.analysis.id
  path_part   = "gladalerts"
}

resource "aws_api_gateway_resource" "summary" {
  rest_api_id = aws_api_gateway_rest_api.raster_analysis_gateway.id
  parent_id   = aws_api_gateway_resource.analysis.id
  path_part   = "summary"
}

resource "aws_api_gateway_method" "treecoverloss_get" {
  rest_api_id   = aws_api_gateway_rest_api.raster_analysis_gateway.id
  resource_id   = aws_api_gateway_resource.treecoverloss.id
  http_method   = "GET"
  authorization = "NONE"
}

resource "aws_api_gateway_method" "gladalerts_get" {
  rest_api_id   = aws_api_gateway_rest_api.raster_analysis_gateway.id
  resource_id   = aws_api_gateway_resource.gladalerts.id
  http_method   = "GET"
  authorization = "NONE"
}

resource "aws_api_gateway_method" "summary_get" {
  rest_api_id   = aws_api_gateway_rest_api.raster_analysis_gateway.id
  resource_id   = aws_api_gateway_resource.summary.id
  http_method   = "GET"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "treecoverloss" {
  rest_api_id = aws_api_gateway_rest_api.raster_analysis_gateway.id
  resource_id = aws_api_gateway_method.treecoverloss_get.resource_id
  http_method = aws_api_gateway_method.treecoverloss_get.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.raster_analysis_gateway.invoke_arn
}

resource "aws_api_gateway_integration" "gladalerts" {
  rest_api_id = aws_api_gateway_rest_api.raster_analysis_gateway.id
  resource_id = aws_api_gateway_method.gladalerts_get.resource_id
  http_method = aws_api_gateway_method.gladalerts_get.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.raster_analysis_gateway.invoke_arn
}

resource "aws_api_gateway_integration" "summary" {
  rest_api_id = aws_api_gateway_rest_api.raster_analysis_gateway.id
  resource_id = aws_api_gateway_method.summary_get.resource_id
  http_method = aws_api_gateway_method.summary_get.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.raster_analysis_gateway.invoke_arn
}

resource "aws_api_gateway_deployment" "raster_analysis_gateway" {
  depends_on = [
    "aws_api_gateway_integration.treecoverloss",
    "aws_api_gateway_integration.gladalerts",
    "aws_api_gateway_integration.summary",
  ]

  rest_api_id = aws_api_gateway_rest_api.raster_analysis_gateway.id
  stage_name  = "default"

  variables = {
    deployed_at = timestamp()
  }
}

resource "aws_lambda_permission" "api_gateway" {
  statement_id  = substr("${local.project}-AllowExecutionFromApiGateway${local.name_suffix}", 0, 64)
  action        = "lambda:InvokeFunction"
  function_name = "${aws_lambda_function.raster_analysis_gateway.function_name}"
  principal     = "apigateway.amazonaws.com"

  # The "/*/*" portion grants access from any method on any resource within the
  # API Gateway REST API.
  source_arn = "${aws_api_gateway_rest_api.raster_analysis_gateway.execution_arn}/*/*"
}