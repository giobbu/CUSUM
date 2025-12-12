Change Point Generator
======================

This module contains the class for generating synthetic data with different types of change points.

Change Point Generator Class
----------------------------

.. autoclass:: source.generator.change_point_generator.ChangePointGenerator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Example Usage
-------------

.. code-block:: python

   import numpy as np
   from source.generator.change_point_generator import ChangePointGenerator

   # Initialize generator
   generator = ChangePointGenerator(num_segments=2, segment_length=1000, change_point_type='gradual_drift')

   # Generate synthetic data
   generator.generate_data()

   # Add a gradual drift change point
   generator.add_gradual_drift(mean_start=10, mean_end=50, std_dev=5, change_point_index=800)

   # Plot generated data
   generator.plot_data()

   # Generate data with random NaNs
   data_with_nans = generator.generate_random_nans(nan_percentage=0.05)
   generator.plot_data_with_nans(data_with_nans)