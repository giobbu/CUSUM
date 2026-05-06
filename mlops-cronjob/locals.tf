locals {
  common_tags = {
    Project     = "mlops-cronjob"
    Environment = "dev"
  }

  naming_prefix = "vpc-a"
}