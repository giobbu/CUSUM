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

### (TODO) Start Kubernetes cluster

Install a local Kubernetes with `minikube` following the [official documentation](https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Fx86-64%2Fstable%2Fbinary+download).

From the terminal start your local cluster: 

```bash
minikube start
```

In a new terminal start Kubernetes dashboard

```bash
minikube dashboard
```

### Install Helm

Install package manager for Kubernetes `helm` following the [official documentation](https://helm.sh/docs/intro/install/)


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
