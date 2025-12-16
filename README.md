[![Python Tests on macOS](https://github.com/giobbu/CUSUM/actions/workflows/python-tests.yml/badge.svg)](https://github.com/giobbu/CUSUM/actions/workflows/python-tests.yml)
![Status](https://img.shields.io/badge/status-development-orange)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14052654.svg)](https://doi.org/10.5281/zenodo.14052654)

# CUSUM

The CUSUM repository contains multiple change point detectors for sequential analysis, enabling the detection of changes in the statistical properties of time-ordered or streaming data.

## Table of Contents

0. [Overview](#0-overview)
1. [CUSUM Detectors](#1-cusum-detectors)
2. [Getting Started](#2-getting-started)
3. [Documentation](#3-documentation)
4. [Example](#5-example)
5. [License](#4-license)

## 0. Overview

A change point is a moment in time at which the statistical properties of a target variable or its underlying data distribution change. Detecting such shifts is critical in domains such as finance, energy markets, healthcare, environmental monitoring, industrial processes, and online advertising, where models and decision-making systems must continuously adapt to evolving conditions.

This project implements several variants of the CUSUM (Cumulative Sum) algorithm for change point detection.

## 1. CUSUM Detectors

CUSUM detectors are sequential algorithms designed to identify change points in time-ordered or streaming data. They process observations incrementally and signal a change when evidence suggests a significant deviation from the expected data distribution.

The implemented algorithms support both:

* **Batch-based detection**, where change points are identified over fixed data windows or batches

* **Instance-based detection**, where each observation is evaluated individually as it arrives

These detectors are therefore suitable for both offline analysis and real-time monitoring scenarios.

## 2. Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/giobbu/CUSUM.git
cd CUSUM
```
 and install dependencies:
```bash
uv sync
```

## 3. Documentation
[![Documentation Status](https://readthedocs.org/projects/CUSUM/badge/?version=latest)](https://CUSUM.readthedocs.io/en/latest/)

Documentation is available at [CUSUM Documentation](https://CUSUM.readthedocs.io/en/latest/)


## 4. Example

The CUSUM detector (or Page-Hinkley test) monitors the cumulative sum of deviations between observed data points and a reference value. When the cumulative sum exceeds a predefined threshold, it signals the presence of a change point.

#### Generate Data
```python
from source.generator.change_point_generator import ChangePointGenerator

# Generate time series data with change points
generator = ChangePointGenerator(num_segments=3, 
                                 segment_length=1000, 
                                 change_point_type='sudden_shift', 
                                 seed=2)
generator.generate_data()
data_stream = generator.get_data()
```

![Image Alt Text](img/readme_data.png)

#### Vanilla CUSUM Detector
```python 
from source.detector.cusum import CUSUM_Detector

# Detect change points using CUSUM Detector
cusum_detector = CUSUM_Detector(warmup_period=500, delta=3, threshold=10)
```

#### a. Instance-based Detection
```python
for data in data_stream:
    pos, neg, is_change = cusum_detector.detection(data)
    print(f"Change Detected: {is_change} \n -Positives: {pos[0]}, \n -Negatives: {neg[0]}")
```

#### b. Batch-based Detection
```python 
# Detect change points using CUSUM Detector
results = cusum_detector.offline_detection(data_stream)

# Plot the detected change points using CUSUM Detector
cusum_detector.plot_change_points(data_stream, 
                                results["change_points"], 
                                results["pos_changes"], 
                                results["neg_changes"])
```

![Image Alt Text](img/readme_cusum.png)


## 5. License
This project is licensed under the GPL-3.0 license - see the [LICENSE](https://github.com/giobbu/CUSUM?tab=GPL-3.0-1-ov-file) file for details.







