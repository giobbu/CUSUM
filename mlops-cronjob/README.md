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

Build Docker image and run the detection task:

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
docker-compose up
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
> - `make local-devup` to install and start services
> - `make local-devdown` to stop services and remove images
---
