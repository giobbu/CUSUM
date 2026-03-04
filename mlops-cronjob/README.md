# Getting started

### Setup virtual environment

```bash
cp -i ../{pyproject.toml,uv.lock} .
```

### Start Kubernetes cluster

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

### Run Docker container

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

### generate synthetic data

create sample data `synthetic_data.csv`:
```bash
cd data
uv run generate_data.py
```

## Manual Offline Detection

To run offline detection from within the /app directory:

```bash
cd /app
uv run detection_task.py
```

Detection results are saved to `/data/detection_results.pkl`.

## Streamlit Visualization

To visualize the results from the CUSUM detector:

```bash
cd /app
uv run streamlit run dashboard.py
```

Open the browser at Streamlit address http://localhost:8501.
The app will read the results from `/data/detection_results.pkl` and display them interactively.