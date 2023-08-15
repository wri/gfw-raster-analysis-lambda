terraform {
  required_providers {
    archive = {
      source = "hashicorp/archive"
    }
    aws = {
      source = "hashicorp/aws"
      version = ">3, <4"
    }
    template = {
      source = "hashicorp/template"
    }
  }
  required_version = ">= 0.13, < 0.14"
}