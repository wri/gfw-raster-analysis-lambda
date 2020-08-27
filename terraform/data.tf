data "template_file" "sts_assume_role_lambda" {
  template = file("policies/sts_assume_role_lambda.json")
}

data "template_file" "raster_analysis_policy" {
  template = file("policies/raster_analysis.json")
}

data "terraform_remote_state" "core" {
  backend = "s3"
  config = {
    bucket = local.tf_state_bucket
    region = "us-east-1"
    key    = "core.tfstate"
  }
}

data "terraform_remote_state" "lambda-layers" {
  backend = "s3"
  config = {
    bucket = local.tf_state_bucket
    region = "us-east-1"
    key    = "lambda-layers.tfstate"
  }
}