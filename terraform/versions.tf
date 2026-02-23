terraform {
  required_providers {
    archive = {
      source = "hashicorp/archive"
    }
    aws = {
      source = "hashicorp/aws"
      version = ">= 5, < 6"
    }
    template = {
      source = "hashicorp/template"
    }
  }
  required_version = "= 0.13.3"
}
