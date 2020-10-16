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