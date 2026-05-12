locals {
  common_tags = {
    Project     = "mlops-cronjob"
    Environment = "dev"
  }

  naming_prefix = "${var.naming_prefix}-${var.environment}"

}