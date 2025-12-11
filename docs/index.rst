CUSUM Documentation
===================

Welcome to the documentation for the CUSUM project.

This site contains:

- A general overview
- Python API documentation generated automatically from your modules

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

API Reference
=============

.. autosummary::
   :toctree: _autosummary
   :recursive:

   source
   source.detector
   source.detector.cusum
   source.detector.mcusum
   source.detector.plot

source.detector.cusum
---------------------

.. autoclass:: source.detector.cusum.CUSUM_Detector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Example Usage
-------------

.. code-block:: python

    import numpy as np
    from source.detector.cusum import CUSUM_Detector

    # Initialize detector
    detector = CUSUM_Detector(warmup_period=10, delta=10, threshold=20)

    # Generate synthetic data
    data = np.concatenate([np.random.normal(0, 1, 100),
                           np.random.normal(5, 1, 100)])

    # Run offline detection
    results = detector.offline_detection(data)

    # Plot change points
    detector.plot_change_points(data,
                                results["change_points"],
                                results["pos_changes"],
                                results["neg_changes"])

