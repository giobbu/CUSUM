# **Concept Drift Detection**

Concept drift refers to the phenomenon where the statistical properties of a target variable or data distribution change over time. Detecting concept drift is crucial in various domains such as financial markets, healthcare, and online advertising to adapt models and decision-making processes to changing environments.

## **Drift Detectors**

Drift detectors are algorithms designed to detect concept drift in streaming data or sequential observations. These detectors analyze the data stream and identify points where the underlying data distribution has changed significantly.

Two commonly used drift detectors are:

### **1. Cumulative Sum (CUSUM) Detector**

The CUSUM detector monitors the cumulative sum of deviations between observed data points and a reference value. When the cumulative sum exceeds a predefined threshold, it signals the presence of a concept drift.

#### Examples

```python
from drift.cusum import CUSUM_Detector

# Initialize CUSUM detector with custom parameters
detector = CUSUM_Detector(warmup_period=10, delta=1, threshold=3)

# Provide sequential data for drift detection
data = np.array([20.3, 18.5, 15.6, 16.8, 15.9, 12.3, 21.8, 22.5, 15, 17.9, 20.2, 25.7, 100.2, 32.5, 32.9, 33.0, 32.2, 31.8, 30.5, 30.1])

# Detect change points in the data
pos_changes, neg_changes, change_points = detector.detect_change_points(data)

# Plot detected change points and cumulative sums
detector.plot_change_points(data, change_points, pos_changes, neg_changes)
```
![Image Alt Text](img/cusum.png)

### **2. Probabilistic Cumulative Sum (CUSUM) Detector**

The Probabilistic CUSUM detector extends the CUSUM method by incorporating statistical probability measures. It evaluates the probability of observing deviations between data points, making it more robust to variations in data distribution.

```python
from drift.cusum import ProbCUSUM_Detector

# Initialize Probabilistic CUSUM detector with custom parameters
detector = ProbCUSUM_Detector(warmup_period=10, threshold_probability=0.001)

# Provide sequential data for drift detection
data = np.array([10.2, 11, 10.5, 12.6, 15.8, 11.9, 13.2, 12.7, 12.5, 12.3, 12.9, 30.0, 21.2, 11.8, 10.5, 10.1])

# Detect change points in the data
probabilities, change_points = detector.detect_change_points(data)

# Plot detected change points and probabilities
detector.plot_change_points(data, change_points, probabilities)

```
![Image Alt Text](img/probcusum.png)