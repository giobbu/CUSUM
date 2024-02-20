# **Change Point Detection**

Change point refers to the phenomenon where the statistical properties of a target variable or data distribution change over time. Detecting change point is crucial in various domains such as financial markets, healthcare, and online advertising to adapt models and decision-making processes to changing environments.

## **Change Point Detectors**

Change point detectors are algorithms designed to detect change points in streaming data or sequential observations. These detectors analyze the data stream and identify points where the underlying data distribution has changed significantly.

Two commonly used drift detectors are:

### **1. Cumulative Sum (CUSUM) Detector**

The CUSUM detector monitors the cumulative sum of deviations between observed data points and a reference value. When the cumulative sum exceeds a predefined threshold, it signals the presence of a change point.

![Image Alt Text](img/cusum.png)

### **2. Probabilistic Cumulative Sum (ProbCUSUM) Detector**

The Probabilistic CUSUM detector extends the CUSUM method by incorporating statistical probability measures. It evaluates the probability of observing deviations between data points, making it more robust to variations in data distribution.

![Image Alt Text](img/probcusum.png)