####################################################
# Create VPC with public and private subnets
####################################################

module "vpc_mlops" {
  source                = "./modules/vpc"
  name                  = "cronjob"
  aws_region            = var.aws_region
  vpc_cidr_block        = var.vpc_cidr_block_a    # "10.1.0.0/16"
  public_subnets_cidrs  = [cidrsubnet(var.vpc_cidr_block_a, 8, 1)]
  private_subnets_cidrs = [cidrsubnet(var.vpc_cidr_block_a, 8, 2)]
  enable_dns_hostnames  = var.enable_dns_hostnames
  aws_azs               = var.aws_azs
  common_tags           = local.common_tags
  naming_prefix         = local.naming_prefix
}
