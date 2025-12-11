Cusum Algorithms
=============

This module contains the core CUSUM change point detection classes.

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


Chart CUSUM Detector
-------------------

.. autoclass:: source.detector.cusum.ChartCUSUM_Detector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Example Usage
-------------

.. code-block:: python

    import numpy as np
    from source.detector.cusum import ChartCUSUM_Detector

    detector = ChartCUSUM_Detector(warmup_period=20, level=3, deviation_type='sqr-dev')
    data = np.concatenate([np.random.normal(0, 1, 100),
                           np.random.normal(5, 1, 100)])
    results = detector.offline_detection(data)
    detector.plot_change_points(data,
                                results["change_points"],
                                results["cusums"],
                                results["upper_limits"],
                                results["lower_limits"])

KS-CUM Detector
---------------

.. autoclass:: source.detector.cusum.KS_CUM_Detector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Example Usage
-------------

.. code-block:: python

    import numpy as np
    from source.detector.cusum import KS_CUM_Detector

    detector = KS_CUM_Detector(window_pre=30, window_post=30, alpha=0.05)
    data = np.concatenate([np.random.normal(0, 1, 100),
                           np.random.normal(5, 1, 100)])
    results = detector.offline_detection(data)
    detector.plot_change_points(data,
                                results["change_points"],
                                results["p_values"])