variable "environment" {
  type        = string
  description = "An environment namespace for the infrastructure."
}

variable "lambda_raster_analysis_runtime" {
  type        = string
  default     = "python3.10"
  description = "Runtime version for AWS Lambda"
}

variable "lambda_tiled_analysis_runtime" {
  type        = string
  default     = "python3.10"
  description = "Runtime version for AWS Lambda"
}

variable "lambda_fanout_runtime" {
  type        = string
  default     = "python3.10"
  description = "Runtime version for AWS Lambda"
}

variable "lambda_raster_analysis_memory_size" {
  type        = number
  default     = 1024
  description = "Memory size version for AWS Lambda"
}

variable "lambda_tiled_analysis_memory_size" {
  type        = number
  default     = 3008
  description = "Memory size version for AWS Lambda"
}

variable "lambda_fanout_memory_size" {
  type        = number
  default     = 128
  description = "Memory size version for AWS Lambda"
}

variable "lambda_raster_analysis_timeout" {
  type        = number
  default     = 60
  description = "Timeout version for AWS Lambda"
}

variable "lambda_tiled_analysis_timeout" {
  type        = number
  default     = 60
  description = "Timeout version for AWS Lambda"
}

variable "lambda_fanout_timeout" {
  type        = number
  default     = 60
  description = "Timeout version for AWS Lambda"
}