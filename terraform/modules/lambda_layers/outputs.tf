output "raster_analysis_arn" {
  value       = aws_lambda_layer_version.raster_analysis.arn
  description = "ARN of raster_analysis lambda layer"
}