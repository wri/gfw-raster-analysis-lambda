variable "environment" {
  type        = string
  description = "An environment namespace for the infrastructure."
}

variable "lambda_raster_analysis_runtime" {
  type        = string
  description = "Runtime version for AWS Lambda"
}

variable "lambda_tiled_analysis_runtime" {
  type        = string
  description = "Runtime version for AWS Lambda"
}

variable "lambda_raster_analysis_gateway_runtime" {
  type        = string
  description = "Runtime version for AWS Lambda"
}

variable "lambda_raster_analysis_memory_size" {
  type        = string
  description = "Memory size version for AWS Lambda"
}

variable "lambda_tiled_analysis_memory_size" {
  type        = string
  description = "Memory size version for AWS Lambda"
}

variable "lambda_raster_analysis_gateway_memory_size" {
  type        = string
  description = "Memory size version for AWS Lambda"
}

variable "lambda_raster_analysis_timeout" {
  type        = string
  description = "Timeout version for AWS Lambda"
}

variable "lambda_tiled_analysis_timeout" {
  type        = string
  description = "Timeout version for AWS Lambda"
}

variable "lambda_raster_analysis_gateway_timeout" {
  type        = string
  description = "Timeout version for AWS Lambda"
}