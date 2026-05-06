variable "aws_region" {
  type        = string
  description = "AWS region to use for resources."
  default     = "eu-central-1"
}

variable "aws_azs" {
  type        = list(string)
  description = "AWS Availability Zones"
  default     = ["eu-central-1a", "eu-central-1b", "eu-central-1c", "eu-central-1d"]
}

variable "enable_dns_hostnames" {
  type        = bool
  description = "Enable DNS hostnames in VPC"
  default     = true
}

variable "vpc_cidr_block_a" {
  type        = string
  description = "Base CIDR Block for VPC A"
  default     = "10.1.0.0/16"
}

variable "vpc_cidr_block_b" {
  type        = string
  description = "Base CIDR Block for VPC B"
  default     = "10.2.0.0/16"
}

variable "vpc_public_subnets_cidr_block" {
  type        = list(string)
  description = "CIDR Block for Public Subnets in VPC"
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24", "10.0.104.0/24"]
}

variable "instance_type" {
  type        = string
  description = "Type for EC2 Instance"
  default     = "t2.micro"
}

variable "sg_ingress_public" {
  type = list(object({
    description = string
    port        = number
  }))
  default = [
    {
      description = "Allows SSH access"
      port        = 22
    },
    {
      description = "Allows HTTP traffic"
      port        = 80
    },
  ]
}

variable "sg_ingress_private" {
  type = list(object({
    description = string
    port        = number
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
  default     = "LAB"
}

variable "environment" {
  type        = string
  description = "Environment for deployment"
  default     = "dev"
}

variable "instance_key" {
  default = "WorkshopKeyPair"
}