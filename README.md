[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![pytest](https://github.com/giobbu/CUSUM/actions/workflows/ci.yml/badge.svg)](https://github.com/giobbu/CUSUM/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/giobbu/CUSUM/branch/main/graph/badge.svg)](https://codecov.io/gh/giobbu/CUSUM)
![Status](https://img.shields.io/badge/status-development-orange)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14052654.svg)](https://doi.org/10.5281/zenodo.14052654)

# CUSUM

The CUSUM repository contains multiple change point detectors for sequential analysis, enabling the detection of changes in the statistical properties of time-ordered or streaming data.

## Table of Contents

0. [Overview](#0-overview)
1. [CUSUM Detectors](#1-cusum-detectors)
2. [Getting Started](#2-getting-started)
3. [Documentation](#3-documentation)
4. [Example: ML Model Performance Monitoring](#4-example:-ml-model-performance-monitoring)
5. [License](#5-license)

## 0. Overview

A change point is a point in time at which the statistical properties of a target variable—or its underlying data-generating process—undergo a significant shift. Detecting such changes is critical in domains such as finance, energy markets, healthcare, environmental monitoring, industrial processes, and online advertising. In these settings, predictive models and decision-making systems must adapt continuously to non-stationary and evolving conditions.

This project implements multiple variants of the CUSUM algorithm for change point detection, enabling robust identification of distributional shifts in sequential data.

## 1. CUSUM Detectors

CUSUM-based detectors are sequential algorithms designed to identify shifts in time-ordered or streaming data. They operate by incrementally processing observations—either instance by instance or in batches—and signaling a change when accumulated evidence indicates a statistically significant deviation from the expected behavior.

The implemented detectors support both:

* **Batch-based detection**, where change points are identified over fixed-size windows or data batches

* **Instance-based detection**, where each incoming observation is evaluated upon arrival

These detectors are, therefore, well suitable for both offline analysis and real-time monitoring in streaming environmnets.

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


## 4. Example: ML Model Performance Monitoring

Performance Monitoring of an instance-based linear learning model applying the CUSUM algorithm.
At each time step:

* Generate a prediction with recursive least squares (RLS) model;
* Acquire the true observed value (if you are lucky);
* Compute residual;
* Apply the CUSUM detector on the residuals to identify potential change points;
* Update the model parameters with the new data instance.

![Image Alt Text](img/monitoring.gif)

## 5. License
This project is licensed under the GPL-3.0 license - see the [LICENSE](https://github.com/giobbu/CUSUM?tab=GPL-3.0-1-ov-file) file for details.







