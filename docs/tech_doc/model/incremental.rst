Instance-based Linear Learning Model
====================================

This module contains the implementation of the Recursive Least Squares algorithm for online linear regression.

Recursive Least Squares Class
-----------------------------
.. autoclass:: source.model.incremental.RecursiveLeastSquares
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__


Example Usage
-------------
.. code-block:: python

      from source.model.incremental import RecursiveLeastSquares
   
      # Initialize model
      model = RecursiveLeastSquares(num_features=3, lambda_factor=0.99, delta=1.0)
   
      # Simulate streaming data
      for _ in range(100):
         X_new = np.random.rand(1, 3)  # New feature vector
         y_new = np.random.rand().reshape(-1, 1)  # New target value
   
         # Make prediction
         y_pred = model.predict(X_new)
         print(f"Predicted: {y_pred}, Actual: {y_new}")

         # Update model with new data
         model.update(X_new, y_new)
         print(f"Updated Weights: {model.weights}")