terraform {
  required_version = ">=0.12.13"
  backend "s3" {
    key     = "wri__gfw-raster-analysis-lambda.tfstate"
    region  = "us-east-1"
    encrypt = true
  }
}

# Download any stable version in AWS provider of 2.36.0 or higher in 2.36 train
provider "aws" {
  region  = "us-east-1"
  version = "~> 2.36.0"
}

module "lambda_layers" {
  source      = "./modules/lambda_layers"
  s3_bucket   = local.tf_state_bucket
  project     = local.project
  name_suffix = local.name_suffix
}