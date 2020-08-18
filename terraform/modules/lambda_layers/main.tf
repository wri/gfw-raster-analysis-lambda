resource "aws_s3_bucket_object" "raster_analysis" {
  bucket = var.s3_bucket
  key    = "lambda_layers/raster_analysis.zip"
  source = "../docker/raster_analysis/layer.zip"
  etag   = filemd5("../docker/raster_analysis/layer.zip")
}

resource "aws_lambda_layer_version" "raster_analysis" {
  layer_name          = substr("${var.project}-raster_analysis${var.name_suffix}", 0, 64)
  s3_bucket           = aws_s3_bucket_object.raster_analysis.bucket
  s3_key              = aws_s3_bucket_object.raster_analysis.key
  compatible_runtimes = ["python3.7"]
  source_code_hash    = "${filebase64sha256("../docker/raster_analysis/layer.zip")}"
}

