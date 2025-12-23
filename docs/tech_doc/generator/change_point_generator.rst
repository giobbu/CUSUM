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

   from source.generator.change_point_generator import ChangePointGenerator

   # Initialize generator
   generator = ChangePointGenerator(num_segments=2, segment_length=1000, change_point_type='gradual_drift')

   # Generate synthetic data
   generator.generate_data()

   # Plot generated data
   generator.plot_data()

**Plotting**

.. image:: ../../_static/images/generator.png
   :alt: Data Generator
   :align: center
   :width: 600px