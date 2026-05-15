# Scheduled Job for Change Point Detection

Change point CUSUM detection algorithms can be run as scheduled job triggered every day, hour, or minute dependening on the need.

## Table of Contents

0. [Overview](#0-overview)
1. [Makefile Commands](#1-makefile-commands)
2. [Getting Started](#2-getting-started)
    - 2.0. [Synthetic Dataset](#20-synthetic-dataset)
    - 2.1. [Python App](#21-python-app)
    - 2.2. [Backend and Frontend Docker Containers](#22-backend-and-frontend-docker-containers)
    - 2.3. [Kubernets CronJob](#23-kubernetes-cronjob)
3. [Local Development](#3-local-development)
4. [AWS EC2 PoC in Terraform](#4-aws-ec2-poc-in-terraform)


## 0. Overview

* Run the CUSUM detection task on data stored as `/data/synthetic_data.csv`.

* The script saves detection results to `/data/<date>/detection_results.pkl`.

* Launch the `Streamlit` dashboard to visualize detection results.

## 1. Makefile Commands

This project includes a Makefile to simplify common development and deployment tasks for the MLOps cronjob system.

Available Commands:

* `make help`
Displays a list of all available commands with short descriptions.

* `make sync`
Installs project dependencies using uv sync.

* `make docker`
Opens/starts the Docker daemon (macOS).

* `make backend-up`
Builds the backend Docker image (detection-cronjob-backend) and runs it in a container with the current project directory mounted.

* `make frontend-up`
Builds the frontend Docker image (streamlit-frontend) and runs it, exposing the app on port 8501.

* `make backend-down`
Removes backend-related Docker images (forcefully, if necessary).

* `make frontend-down`
Stops and removes the frontend container and deletes its Docker image.


## 2. Getting Started

### 2.0. Synthetic Dataset

### Setup virtual environment

```bash
cp -i ../{pyproject.toml,uv.lock} .
```

### Add cusum-based detectors and data generation scripts

* copy `source/detector` in `/backend`

```bash
mkdir backend/detector
cp -i -r ../source/detector/* backend/detector
```
* copy `source/generator` in `/data`

```bash
mkdir data/generator
cp -i -r ../source/generator/* ./data/generator
```

### Generate synthetic data

create sample data `synthetic_data.csv`:

```bash
uv run data/generate_data.py
```


## 2.1. Python App

### CUSUM Detection
You can run the detection task manually from the backend directory:

```bash
uv run backend/detection_task.py
```

This will:

* Load the collected data from `/data/synthetic_data.csv`
* Run the CUSUM change point detection
* The results are saved to `/data/<date>/detection_results.pkl`

### Streamlit Dashboard

Launch dashboard:

```bash
uv run streamlit run frontend/dashboard.py
```

Open browser at `http://localhost:8501`

The dashboard will display the most recent detection results.


## 2.2. Backend and Frontend Docker Containers

### Backend

Build Docker image and run the detection task (files located in `quick-start`):

```bash
docker build -f dockerfile.backend.dev -t detection-cronjob-backend-dev .
docker run -d -v "$PWD":/home -it detection-cronjob-backend-dev
```
This runs the detection process in the background mounting the current directory.

### Frontend

Build and start streamlit dashboard:

```bash
docker build -f dockerfile.frontend.dev -t dashboard-dev .
docker run -p 8501:8501 -v "$PWD":/home -it dashboard-dev
```
---
> ⚠️ **Makefile Commands**
>
> - `make backend-up` to build and start backend
> - `make frontend-up` to build and start frontend
> 
> - `make backend-down` to remove backend
> - `make frontend-down` to remove frontend
---

## 2.3. Kubernetes Cronjob

This option runs the detection pipeline as a scheduled Kubernetes CronJob using a local cluster powered by `minikube`.

### Prerequisites
Install the following tools:

* **Minikube – local Kubernetes cluster** - installation guide: https://minikube.sigs.k8s.io/docs/start/
* **Helm – Kubernetes package manager** - installation guide: https://helm.sh/docs/intro/install/

### Kubernetes Local Cluster

Start `minikube`: 

```bash
minikube start
```
### Mount Volume

In a new terminal inside Minikube, to access local project files, mount the current project directory:

```bash
minikube mount "$(pwd)":/host
```

### Configure Docker to use Minikube

Return to the first terminal and configure Docker to build images directly inside the Minikube environment:

```bash
eval $(minikube docker-env)
```

Build docker image:

```bash
docker build -t detection-cronjob:1.0 .
```

### Install CronJob with Helm

Install Helm chart:

```bash
helm install detection detection-cronjob
```

Verify the deployment:

```bash
helm list
```

Example output:

```bash
NAME            NAMESPACE       REVISION        UPDATED                                 STATUS          CHART                 APP VERSION
detection       default         1               2026-03-06 18:12:29.287216 +0100 CET    deployed        detection-cronjob-0.1.0 1.16.0
```

Verify cronjob:

```bash
kubectl get cronjobs

NAME        SCHEDULE      SUSPEND   ACTIVE   LAST SCHEDULE
detection   */1 * * * *   False     0        <time>
```

### Debugging

test the job immediatelly:

```bash
kubectl create job --from=cronjob/detection test-run
```

```bash
kubectl get pods --watch
```

```bash
kubectl logs <pod-name>
```

Delet all jobs and pods:

```bash
kubectl delete jobs --all                                             
kubectl delete pods --all
```

Suspend cronjob:

```bash
kubectl patch cronjob detection-cronjob -p '{"spec":{"suspend":true}}'
```

## 3. Local Development

Use Docker Compose to spin up the full microservices stack locally:

```bash
docker-compose -f docker-compose.local.yaml up -d
```

```bash
docker image list
REPOSITORY           TAG       IMAGE ID       CREATED        SIZE
streamlit-frontend   latest    9643432c64e6   29 hours ago   1.21GB
detection-backend    latest    248c223578b7   46 hours ago   1.21GB
apache/airflow       2.9.0     2b0695195cf0   2 years ago    2.01GB
```
Once all services are running, you can access:

* Streamlit dashboard → http://localhost:8501
* Airflow UI → http://localhost:8080

The first startup may take a few minutes as images are built and services initialize.

To stop and remove all running services:
```bash
docker compose down --rmi all
```

---
> ⚠️ **Makefile Commands**
>
> - `make local-up` to install and start services
> - `make local-down` to stop services and remove images
---


## 4. AWS EC2 PoC in Terraform

`cd terraform` and creates the aws resources:

* networking
    1. VPC
    2. IGW
    3. private and public subnets
    4. private and public route tables
    5. public route table association

* instances
    1. bastion host and private ec2
    2. public and private security groups
    3. NAT instance

* registry
    1. ECR
    2. Instance Profile for EC2 IAM role access


run the following 

```bash
terraform init
terraform plan

# create everything including NAT
terraform apply -var="enable_nat=true"
```
Outputs the public and private ip addresses `terraform/outputs.tf`.

Connect to the private instance
```bash
# change permissions
chmod 400 CronKeyPair.pem
# connect to ec2
ssh -i CronKeyPair.pem -o ProxyCommand="ssh -i CronKeyPair.pem -W %h:%p ec2-user@$(terraform output -raw bastion_host_ip)" ec2-user@$(terraform output -raw private_ec2_ip)
```

Copy docker-compose file to EC2
```bash
cd Terraform
BASTION_IP=$(terraform output -raw bastion_host_ip)
PRIVATE_IP=$(terraform output -raw private_ec2_ip)
cd ..

scp -r -i Terraform/CronKeyPair.pem \
  -o ProxyCommand="ssh -i Terraform/CronKeyPair.pem -W %h:%p ec2-user@$BASTION_IP" \
  data/ .env dockerfile.backend dockerfile.frontend docker-compose.aws.yaml nginx.conf airflow-init.sh \
  ec2-user@$PRIVATE_IP:~/
```

Install Docker
```bash
sudo yum update -y
sudo yum install docker -y
sudo systemctl start docker
sudo systemctl enable docker

# install docker-compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose

sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

# verify installation
docker-compose --version
```

Build and Push to ECR from local machine
```bash
#login
aws ecr get-login-password --region <aws_zone> | sudo docker login --username AWS --password-stdin AWS_ACCOUNT_ID.dkr.ecr.<aws_zone>.amazonaws.com

# build images locally
docker build -t detection-backend:latest -f dockerfile.backend .
docker build -t streamlit-frontend:latest -f dockerfile.frontend .

# tag for ECR
docker tag detection-backend:latest \
  <aws_account_id>.dkr.ecr.<aws_zone>.amazonaws.com/cusum-repo:detection-backend
docker tag streamlit-frontend:latest \
  <aws_account_id>.dkr.ecr.<aws_zone>.amazonaws.com/cusum-repo:streamlit-frontend

# push to ECR
docker push <aws_account_id>.dkr.ecr.<aws_zone>.amazonaws.com/cusum-repo:detection-backend
docker push <aws_account_id>.dkr.ecr.<aws_zone>.amazonaws.com/cusum-repo:streamlit-frontend

# list images
aws ecr list-images --repository-name cusum-repo --region <aws_zone>
repo --region <aws_zone>
{
    "imageIds": [
        {
            "imageTag": "detection-backend", 
            "imageDigest": "sha256:2f6318251ff5bd400686875156dd66c7199077adc89553d3e0bf90c0875109d1"
        }, 
        {
            "imageTag": "streamlit-frontend", 
            "imageDigest": "sha256:395a8181ba0bd1aece389a5271ac7aa54fde167aaba9288b304da955f999842a"
        }
    ]
}
```

Spin up services
```bash
sudo docker-compose -f docker-compose.aws.yaml up -d
```

Disable NAT
```bash
# destroy only NAT
terraform apply -var="enable_nat=false"
```

Port forwarding
```bash
ssh -i CronKeyPair.pem -L 8080:$(terraform output -raw private_ec2_ip):80 -o ProxyCommand="ssh -i CronKeyPair.pem -W %h:%p ec2-user@$(terraform output -raw bastion_host_ip)" ec2-use
```
Open `localhost:8080`

Teardown infrastructure
```bash
terraform destroy -var="enable_nat={true,false}"
```





