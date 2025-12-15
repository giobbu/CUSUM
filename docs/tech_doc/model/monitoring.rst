ML Model Performance Monitoring
===============================

This section covers how to apply CUSUM detection algorithms for monitoring the performance drifts of online ML models in real-time.

.. note::
    See the `Notebook <https://github.com/giobbu/CUSUM/blob/main/notebooks/example_ml_monitoring.ipynb>`_ for code implementation details.

Model Monitoring with CUSUM
----------------------------
Steps to monitor an online autoregressive linear model using the CUSUM algorithm:

1. Generate a prediction with recursive least squares (RLS) model.
2. Retrieve the true observed value.
3. Update the model parameters with the new data instance.
4. Compute residual.
5. Apply the CUSUM detector on the residuals to identify potential change points.

.. image:: ../../_static/images/monitoring.gif
   :alt: Model Monitoring Workflow
   :align: center
   :width: 600px