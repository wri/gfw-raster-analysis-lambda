resource "aws_iam_policy" "raster_analysis" {
  name   = substr("${local.project}-${local.name_suffix}", 0, 64)
  path   = "/"
  policy = data.template_file.raster_analysis_policy.rendered
}

resource "aws_iam_role" "raster_analysis_lambda" {
  name               = substr("${local.project}-lambda${local.name_suffix}", 0, 64)
  assume_role_policy = data.template_file.sts_assume_role_lambda.rendered
}

resource "aws_iam_role_policy_attachment" "raster_analysis_lambda" {
  role       = aws_iam_role.raster_analysis_lambda.name
  policy_arn = aws_iam_policy.raster_analysis.arn
}

