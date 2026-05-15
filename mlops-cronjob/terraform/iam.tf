####################################################
# IAM Role for EC2
####################################################
resource "aws_iam_role" "ec2_role" {
  name = "${local.naming_prefix}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "ec2.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}

####################################################
# Attach ECR Policy to Role
####################################################
resource "aws_iam_role_policy_attachment" "ecr_access" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

####################################################
# Instance Profile
####################################################
resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${local.naming_prefix}-ec2-profile"
  role = aws_iam_role.ec2_role.name
}