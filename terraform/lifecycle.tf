resource "aws_s3_bucket_lifecycle_configuration" "example" {
  bucket = data.terraform_remote_state.core.outputs.pipelines_bucket

  rule {
    id = "expire-jobs"

    filter {
      prefix = "analysis/jobs/"
    }

    expiration {
      days = 90
    }

    status = "Enabled"
  }
}