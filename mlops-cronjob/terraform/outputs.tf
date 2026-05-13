output "bastion_host_ip" {
  value = module.public_bastion.public_ip
}

output "private_ec2_ip" {
  value = module.private_instance.private_ip
}