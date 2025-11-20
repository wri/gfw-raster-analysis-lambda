terraform {

  backend "s3" {
    key     = "wri__gfw-raster-analysis-lambda.tfstate"
    region  = "us-east-1"
    encrypt = true
  }
}

module "lambda_layers" {
  source      = "./modules/lambda_layers"
  s3_bucket   = local.tf_state_bucket
  project     = local.project
  name_suffix = local.name_suffix
}

module "ssm" {
  source      = "git::https://github.com/wri/gfw-terraform-modules.git//terraform/modules/ssm?ref=v0.4.2.8"
  environment = var.environment
  namespace   = "gfw-lambda-layers"
  contract = {
    raster_analysis_state_machine_arn = aws_sfn_state_machine.process_list.arn
    raster_analysis_lambda_arn        = aws_lambda_function.tiled_raster_analysis.arn
    raster_analysis_lambda_name       = aws_lambda_function.tiled_raster_analysis.function_name
  }
  lists = {}
  strings = {}
  secure_strings = {}
}
