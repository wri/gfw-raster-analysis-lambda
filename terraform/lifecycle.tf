resource "aws_s3_bucket_lifecycle_configuration" "example" {
  bucket = local.core.gfw_pipelines_bucket_name

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