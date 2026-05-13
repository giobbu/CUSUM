output "instance_id" {
  value = aws_instance.ec2.id
}

output "security_group_id"{
  value = aws_security_group.security_group.id
}

output "public_ip"{
  value = aws_instance.ec2.public_ip
}

output "private_ip"{
  value = aws_instance.ec2.private_ip
}