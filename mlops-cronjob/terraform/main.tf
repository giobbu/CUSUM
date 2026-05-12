####################################################
# Create VPC with public and private subnets
####################################################
module "vpc_mlops" {
  source                = "./modules/network"
  name                  = "cronjob"
  aws_region            = var.aws_region
  vpc_cidr_block        = var.vpc_cidr_block
  public_subnets_cidrs_block  = [cidrsubnet(var.vpc_cidr_block, 8, 1)]
  private_subnets_cidrs_block = [cidrsubnet(var.vpc_cidr_block, 8, 2)]
  aws_azs               = var.aws_azs
  enable_dns_hostnames  = var.enable_dns_hostnames
  enable_dns_support    = var.enable_dns_support
  common_tags           = local.common_tags
  naming_prefix         = local.naming_prefix
}
