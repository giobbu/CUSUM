CUSUM Project
===================

Overview
--------

A change point is a moment in time at which the statistical properties of a target variable or its underlying data distribution change. Detecting such shifts is critical in domains such as finance, energy markets, healthcare, environmental monitoring, industrial processes, and online advertising, where models and decision-making systems must continuously adapt to evolving conditions.

This project implements several variants of the CUSUM (Cumulative Sum) algorithm for change point detection.

CUSUM Detectors
----------------

CUSUM detectors are sequential algorithms designed to identify change points in time-ordered or streaming data. They process observations incrementally and signal a change when evidence suggests a significant deviation from the expected data distribution.

The implemented algorithms support both:

* **Batch-based detection**, where change points are identified over fixed data windows or batches

* **Instance-based detection**, where each observation is evaluated individually as it arrives

These detectors are therefore suitable for both offline analysis and real-time monitoring scenarios.