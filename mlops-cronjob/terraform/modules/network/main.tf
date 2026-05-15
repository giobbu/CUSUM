####################################################
#           Create the VPC
####################################################
resource "aws_vpc" "app_vpc" {
  cidr_block           = var.vpc_cidr_block

  tags = merge(var.common_tags, {
    Name = "${var.naming_prefix}-${var.name}"
  })
}


####################################################
#           Create the internet gateway
####################################################
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.app_vpc.id

  tags = merge(var.common_tags, {
    Name = "${var.naming_prefix}-igw"
  })
}


####################################################
#           Create the public subnets
####################################################
resource "aws_subnet" "public_subnets" {

  for_each          = toset(var.public_subnets_cidrs_block)
  vpc_id            = aws_vpc.app_vpc.id
  cidr_block        = each.value
  availability_zone =  var.aws_azs[index(var.public_subnets_cidrs_block, each.value)]


  map_public_ip_on_launch = true  # This makes public subnet

  tags = merge(var.common_tags, {
    Name = "${var.naming_prefix}-pub_subnet-${each.key}"
  })
}



####################################################
#           Create the private subnets
####################################################
resource "aws_subnet" "private_subnets" {
  
  for_each          = toset(var.private_subnets_cidrs_block)
  vpc_id            = aws_vpc.app_vpc.id
  cidr_block        = each.value
  availability_zone = var.aws_azs[index(var.private_subnets_cidrs_block, each.value)]

  map_public_ip_on_launch = false  # This makes private subnet

  tags = merge(var.common_tags, {
    Name = "${var.naming_prefix}-priv_subnet-${each.key}"
  })
}


####################################################
# Create the public route table
####################################################
resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.app_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = merge(var.common_tags, {
    Name = "${var.naming_prefix}-pub-rt_tbl"
  })
}


####################################################
# Assign the public route table to the public subnet
####################################################
resource "aws_route_table_association" "public_rt_asso" {
  for_each       = toset(var.public_subnets_cidrs_block)
  subnet_id      = aws_subnet.public_subnets[each.key].id
  route_table_id = aws_route_table.public_rt.id
}




####################################################
# Set Elastic IP
####################################################
resource "aws_eip" "nat" {
  count = var.enable_nat ? 1 : 0
  vpc   = true
}

####################################################
# Set NAT instance
####################################################
resource "aws_nat_gateway" "nat" {
  count         = var.enable_nat ? 1 : 0
  allocation_id = aws_eip.nat[0].id
  subnet_id     = values(aws_subnet.public_subnets)[0].id
}

####################################################
# Set default route table as private route table
####################################################
resource "aws_default_route_table" "private_route_table" {
  default_route_table_id = aws_vpc.app_vpc.default_route_table_id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat[0].id
  }
}