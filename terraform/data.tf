data "template_file" "sts_assume_role_lambda" {
  template = file("policies/sts_assume_role_lambda.json")
}

data "template_file" "raster_analysis_policy" {
  template = file("policies/raster_analysis.json")
}

data "template_file" "sts_assume_role_states" {
  template = file("policies/sts_assume_role_states.json")
}

data "template_file" "sfn_process_list" {
  template = file("../step_functions/process_list.json.tmpl")
  vars = {
    lambda_preprocessing_name = aws_lambda_function.preprocessing.function_name
    lambda_list_tiled_raster_analysis_name = aws_lambda_function.list_tiled_raster_analysis.function_name
    lambda_aggregation_name = aws_lambda_function.aggregation.function_name
  }
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