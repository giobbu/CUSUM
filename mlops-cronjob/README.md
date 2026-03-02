# Getting started

### setup virtual environment

```bash
cp -i ../{pyproject.toml,uv.lock} .
```

### install minikube for local cluster

### pull image from docker registry

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