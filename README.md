[![Python Tests on macOS](https://github.com/giobbu/CUSUM/actions/workflows/python-tests.yml/badge.svg)](https://github.com/giobbu/CUSUM/actions/workflows/python-tests.yml)
![Status](https://img.shields.io/badge/status-development-orange)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14052654.svg)](https://doi.org/10.5281/zenodo.14052654)


## Overview

A change point refers to a moment in time when the statistical properties of a target variable or data distribution shift. Detecting these changes is essential in domains such as finance, energy markets, healthcare, environmental monitoring, industrial processes, and online advertising, where models and decisions must continually adapt to evolving conditions. 


## **CUSUM Detectors**

CUSUM detectors are algorithms designed to identify change points in streaming data or sequential observations. These detectors analyze the data instance by instance and identify points where the underlying data distribution has changed significantly.

The algorithms can be applied to both data batches, **batch-based detection**, and individual data instances, **instance-based detection**. The algorithms implemented here are suitable for use in both settings.

### Example - **CUSUM Detector**

The CUSUM detector (or Page-Hinkley test) monitors the cumulative sum of deviations between observed data points and a reference value. When the cumulative sum exceeds a predefined threshold, it signals the presence of a change point.

```python 
# Detect change points using CUSUM Detector
cusum_detector = CUSUM_Detector(warmup_period=500, delta=3, threshold=10)
```

#### Instance-based Detection
```python 
for data in generator.data:
    pos, neg, is_change = cusum_detector.detection(data)
    print(f"Change Detected: {is_change} \n -Positives: {pos[0]}, \n -Negatives: {neg[0]}")
```

#### Batch-based Detection
```python 
# Detect change points using CUSUM Detector
results = cusum_detector.offline_detection(generator.data)

# Plot the detected change points using CUSUM Detector
cusum_detector.plot_change_points(generator.data, 
                                results["change_points"], 
                                results["pos_changes"], 
                                results["neg_changes"])
```

![Image Alt Text](img/cusum.png)


## Documentation
[![Documentation Status](https://readthedocs.org/projects/CUSUM/badge/?version=latest)](https://CUSUM.readthedocs.io/en/latest/)

Documentation is available at [CUSUM Documentation](https://CUSUM.readthedocs.io/en/latest/)

## License
This project is licensed under the GPL-3.0 license - see the [LICENSE](https://github.com/giobbu/CUSUM?tab=GPL-3.0-1-ov-file) file for details.




 
