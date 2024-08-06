resource "aws_sfn_state_machine" "process_list" {
  name       = substr("${local.project}-process_list${local.name_suffix}", 0, 64)
  role_arn   = aws_iam_role.process_list-states.arn
  definition = data.template_file.sfn_process_list.rendered
  tags       = local.tags
}