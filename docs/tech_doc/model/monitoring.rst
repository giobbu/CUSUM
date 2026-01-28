Performance Monitoring with CUSUM
======================================

This section covers how to apply CUSUM detection algorithms for monitoring the performance drifts of instance-based learning models in real-time.

.. note::
    See the `Notebook <https://github.com/giobbu/CUSUM/blob/main/notebooks/notebook_rls_monitoring.ipynb>`_ for code implementation details.

Performance Monitoring
----------------------------
Steps to monitor an instance-based model using the CUSUM algorithm:

1. Generate a prediction with recursive least squares (RLS) model.
2. Retrieve the true observed value.
3. Compute residual.
4. Apply the CUSUM detector on the residuals to identify potential change points.
5. Update the model parameters with the new data instance.

.. image:: ../../_static/images/monitoring.gif
   :alt: Model Monitoring with CUSUM
   :align: center
   :width: 600px