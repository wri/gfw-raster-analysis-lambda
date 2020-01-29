output "lambda_raster_analysis" {
  value = aws_lambda_function.raster_analysis.id
}

output "lambda_tiled_analysis" {
  value = aws_lambda_function.tiled_analysis.id
}

output "lambda_raster_analysis_gateway" {
  value = aws_lambda_function.raster_analysis_gateway.id
}