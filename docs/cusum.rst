CUSUM Classes
=============

This file documents all classes in `source.detector.cusum`.

CUSUM Detector Class
-------------------

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

    detector = CUSUM_Detector(warmup_period=10, delta=10, threshold=20)
    data = np.concatenate([np.random.normal(0, 1, 100),
                           np.random.normal(5, 1, 100)])
    results = detector.offline_detection(data)
    detector.plot_change_points(data,
                                results["change_points"],
                                results["pos_changes"],
                                results["neg_changes"])

Probabilistic CUSUM Detector Class
----------------------------------

.. autoclass:: source.detector.cusum.ProbCUSUM_Detector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Example Usage
-------------

.. code-block:: python

    import numpy as np
    from source.detector.cusum import ProbCUSUM_Detector

    detector = ProbCUSUM_Detector(warmup_period=10, threshold_probability=0.01)
    data = np.concatenate([np.random.normal(0, 1, 100),
                           np.random.normal(5, 1, 100)])
    results = detector.offline_detection(data)
    detector.plot_change_points(data,
                                results["change_points"],
                                results["probabilities"])