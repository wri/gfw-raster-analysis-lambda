resource "aws_dynamodb_table" "tiled_results_table" {
  name           = substr("${local.project}-tiled_results${local.name_suffix}", 0, 64)
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "tile_id"
  range_key      = "part_id"

  attribute {
    name = "tile_id"
    type = "S"
  }

  attribute {
    name = "part_id"
    type = "N"
  }

  ttl {
    attribute_name = "time_to_live"
    enabled        = true
  }
}

resource "aws_dynamodb_table" "tiled_status_table" {
  name           = substr("${local.project}-tiled_status${local.name_suffix}", 0, 64)
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "tile_id"

  attribute {
    name = "tile_id"
    type = "S"
  }

  ttl {
    attribute_name = "time_to_live"
    enabled        = true
  }
}