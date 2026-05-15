variable "name" {}
variable "aws_region" {}
variable "aws_azs" {}
variable "enable_dns_hostnames" {}
variable "enable_dns_support" {}
variable "vpc_cidr_block" {}
variable "public_subnets_cidrs_block" {}
variable "private_subnets_cidrs_block" {}
variable "common_tags" {}
variable "naming_prefix" {}

variable "enable_nat" {
  type    = bool
  default = true
}