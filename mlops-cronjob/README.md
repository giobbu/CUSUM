# Scheduled Job for Change Point Detection

Detection can be run manually or scheduled via a Kubernetes cron job. Results can then be explored through a Streamlit dashboard.

## Workflow Overview

* Run the CUSUM detection task (Python script or scheduled job).

* The script saves detection results to `/data/detection_results.pkl`.

* Launch the `Streamlit` dashboard to visualize the results.

## Getting started

### Setup virtual environment

```bash
cp -i ../{pyproject.toml,uv.lock} .
```

### copy cusum-based detectors and data generation scripts

* copy `source/detector` in `/app`

```bash
mkdir app/detector
cp -i -r ../source/detector/* ./app/detector
```
* copy `source/generator` in `/data`

```bash
mkdir data/generator
cp -i -r ../source/generator/* ./data/generator
```

### Generate synthetic data

create sample data `synthetic_data.csv`:

```bash
cd data
uv run generate_data.py
```

## Running the Detection

### *Option 1* — Run the Python Script

You can run the detection task manually from the /app directory:

```bash
cd /app
uv run detection_task.py
```

This will:

* Load the collected data
* Run the CUSUM change point detection
* Save the results to `/data/detection_results.pkl`

### *Option 2* — Run via Docker

Alternatively, run the detection task inside a Docker container:

```bash
docker run -d -v "$(pwd)":/home -it schedule-detection
```

This runs the detection process in the background and mounts the current directory into the container.

## Streamlit Visualization

The detection results are stored in `/data/detection_results.pkl`

To launch the visualization dashboard:

```bash
cd /app
uv run streamlit run dashboard.py
```

Once started, open your browser at `http://localhost:8501`

The dashboard will load the detection results and display them interactively.


### *Option 3* — Run via Kubernetes Cronjob

This option runs the detection pipeline as a scheduled Kubernetes CronJob using a local cluster powered by `minikube`.

#### Prerequisites
Install the following tools:

* **Minikube – local Kubernetes cluster** - installation guide: https://minikube.sigs.k8s.io/docs/start/
* **Helm – Kubernetes package manager** - installation guide: https://helm.sh/docs/intro/install/

#### Start local Kubernetes cluster

Start `minikube`: 

```bash
minikube start
```
#### Mount project directory

In a new terminal inside Minikube, to access local project files, mount the current project directory:

```bash
minikube mount "$(pwd)":/host
```

#### Configure Docker to use Minikube and build image

Return to the first terminal and configure Docker to build images directly inside the Minikube environment:

```bash
eval $(minikube docker-env)
```

Build docker image:

```bash
docker build -t detection-cronjob:1.0 .
```

#### Install cronjob with Helm

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
NAME            NAMESPACE       REVISION        UPDATED                                 STATUS          CHART                   APP VERSION
detection       default         1               2026-03-06 18:12:29.287216 +0100 CET    deployed        detection-cronjob-0.1.0 1.16.0
```

Verify cronjob:

```bash
kubectl get cronjobs

NAME        SCHEDULE      SUSPEND   ACTIVE   LAST SCHEDULE
detection   */1 * * * *   False     0        <time>
```

#### Debugging

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