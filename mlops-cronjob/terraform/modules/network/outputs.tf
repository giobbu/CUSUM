output "vpc_id" {
  value = aws_vpc.app_vpc.id
}


output "public_subnets" {
  value = values(aws_subnet.public_subnets)[*].id
}

output "private_subnets" {
  value = values(aws_subnet.private_subnets)[*].id
}

output "public_route_table_id" {
  value = aws_route_table.public_rt.id
}

output "private_route_table_id" {
  value = aws_default_route_table.private_route_table.id
}