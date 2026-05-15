####################################################
# Create VPC with public and private subnets
####################################################
module "vpc" {
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
  enable_nat                  = var.enable_nat

}

module "public_bastion"  {
  source           = "./modules/instances"
  instance_type    = var.instance_type
  instance_key     = var.instance_key
  subnet_id        = module.vpc.public_subnets[0]
  vpc_id           = module.vpc.vpc_id
  ec2_name         = "Public Bastion Host"
  sg_ingress_ports = var.sg_ingress_public
  common_tags      = local.common_tags
  naming_prefix    = local.naming_prefix
}

module "private_instance" {
  source           = "./modules/instances"
  instance_type    = var.instance_type
  instance_key     = var.instance_key
  subnet_id        = module.vpc.private_subnets[0]
  vpc_id           = module.vpc.vpc_id
  ec2_name         = "Private EC2"
  sg_ingress_ports = var.sg_ingress_private
  common_tags      = local.common_tags
  naming_prefix    = local.naming_prefix
  instance_profile = aws_iam_instance_profile.ec2_profile.name

}

####################################################
# Amend Private SG to allow traffic from Bastion SG
####################################################

resource "aws_security_group_rule" "private_ssh_from_bastion" {
  type                     = "ingress"
  from_port                = 22
  to_port                  = 22
  protocol                 = "tcp"
  security_group_id        = module.private_instance.security_group_id
  source_security_group_id = module.public_bastion.security_group_id
}

resource "aws_security_group_rule" "private_http_from_bastion" {
  type                     = "ingress"
  from_port                = 80
  to_port                  = 80
  protocol                 = "tcp"
  security_group_id        = module.private_instance.security_group_id
  source_security_group_id = module.public_bastion.security_group_id
}

