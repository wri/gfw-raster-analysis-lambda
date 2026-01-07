locals {
  bucket_suffix   = var.environment == "production" ? "" : "-${var.environment}"
  tf_state_bucket = "gfw-terraform${local.bucket_suffix}"
  name_suffix     = "-${terraform.workspace}"
  project         = "raster-analysis"
  core            = jsondecode(data.aws_ssm_parameter.core_contract.value)
  lambda_layers   = jsondecode(data.aws_ssm_parameter.lambda_layers_contract.value)
  tags            = local.core.tags
}