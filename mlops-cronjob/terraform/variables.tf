####################################################
# VPC with public and private subnets
####################################################

variable "aws_region" {
  type        = string
  description = "AWS region to use for resources."
  default     = "eu-central-1"
}

variable "aws_azs" {
  type        = list(string)
  description = "AWS Availability Zones"
  default     = ["eu-central-1a", "eu-central-1b", "eu-central-1c"]
}

variable "enable_dns_support" {
  type        = bool
  description = "Enable DNS support in VPC"
  default     = true
}

variable "enable_dns_hostnames" {
  type        = bool
  description = "Enable DNS hostnames in VPC"
  default     = true
}

variable "vpc_cidr_block" {
  type        = string
  description = "Base CIDR Block for VPC A"
  default     = "10.0.0.0/16"
}

variable "public_subnets_cidrs_block" {
  type        = list(string)
  description = "CIDR Block for Public Subnets in VPC"
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]  # matches cidrsubnet pattern
}

variable "private_subnets_cidrs_block" {
  type        = list(string)
  description = "CIDR blocks for private subnets"
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

####################################################
# Bastion Host and Private Instances
####################################################

variable "instance_type" {
  type        = string
  description = "Type for EC2 Instance"
  default     = "t3.micro"
}

variable "sg_ingress_public" {
  type = list(object({
    description = string
    port   = number
    protocol    = string
    cidr_blocks = list(string)
  }))

  default = [
    {
      description = "Allow SSH access"
      port   = 22
      protocol    = "tcp"
      cidr_blocks = ["87.13.46.103/32"]
    },
  ]
}


variable "sg_ingress_private" {
  type = list(object({
    description = string
    port        = number
    protocol    = string
    cidr_blocks = list(string)
  }))
  default = []
}

variable "company" {
  type        = string
  description = "Company name for resource tagging"
  default     = "CT"
}

variable "project" {
  type        = string
  description = "Project name for resource tagging"
  default     = "Project"
}

variable "naming_prefix" {
  type        = string
  description = "Naming prefix for all resources."
  default     = "CRONJOB"
}

variable "environment" {
  type        = string
  description = "Environment for deployment"
  default     = "dev"
}

variable "instance_key" {
  default = "CronKeyPair"
}