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


**With Missing Data**


* **Point Missingness**

.. code-block:: python

   # Generate data with point NaNs
   data_with_point_nans = generator.generate_point_nans(percentage=0.4)

   # Plot data with Point NaNs
   generator.plot_data_with_nans(data_with_point_nans)

.. image:: ../../_static/images/generator_point_nans.png
   :alt: Data Generator with Point NaNs
   :align: center
   :width: 600px


* **Block Missingness**

.. code-block:: python

   # Generate data with point NaNs
   data_with_nans = generator.generate_block_nans(percentage=0.1, min_block_size=5, max_block_size=50)

   # Plot data with Block NaNs
   generator.plot_data_with_nans(data_with_nans)

.. image:: ../../_static/images/generator_block_nans.png
   :alt: Data Generator with Block NaNs
   :align: center
   :width: 600px