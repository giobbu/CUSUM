SGD Learning Model
====================================

This module contains the implementation of the Stochastic Gradient Descent algorithm for online linear regression.

Stochastic Gradient Descent Class
---------------------------------
.. autoclass:: source.model.incremental.StochasticGradientDescent
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__


Example Usage
-------------
.. code-block:: python

      from source.model.incremental import StochasticGradientDescent
      import numpy as np
   
      # Initialize model
      model = StochasticGradientDescent(num_features=3, learning_rate=1e-5)
   
      # Simulate streaming data
      for _ in range(100):
      
         # Generate random data
         X_new = np.random.rand(1, 3)  # New feature vector
         y_new = np.random.rand().reshape(-1, 1)  # New target value
   
         # Make prediction
         y_pred = model.predict(X_new)
         print(f"Predicted: {y_pred}, Actual: {y_new}")

         # Update model with new data
         model.update(X_new, y_new)
         print(f"Updated Weights: {model.weights}")