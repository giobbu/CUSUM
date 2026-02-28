SGD-CUSUM
======================================

This section covers how to apply CUSUM detection algorithms for monitoring the performance drifts of instance-based learning models in real-time.


SGD-CUSUM based Monitoring
----------------------------
Steps to monitor an instance-based model using the CUSUM algorithm:

1. Generate a prediction with stochastic gradient descent (SGD) model.
2. Retrieve the true observed value.
3. Compute residual.
4. Apply the CUSUM detector on the residuals to identify potential change points.
5. Update the model parameters with the new data instance.

.. image:: ../../_static/images/sgd_monitoring.gif
   :alt: Model Monitoring with CUSUM
   :align: center
   :width: 600px