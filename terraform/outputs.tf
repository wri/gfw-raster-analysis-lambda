output "raster_analysis_lambda_name" {
  value = aws_lambda_function.tiled_raster_analysis.function_name
}

output "raster_analysis_lambda_arn" {
  value = aws_lambda_function.tiled_raster_analysis.arn
}

output "raster_analysis_state_machine_arn" {
  value = aws_sfn_state_machine.process_list.arn
}