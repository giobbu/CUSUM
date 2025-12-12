Vanilla CUSUM Detector Class
----------------------------

.. autoclass:: source.detector.cusum.CUSUM_Detector
   :members: 
   :undoc-members:  
   :show-inheritance:
   :special-members: __init__

Example 
-------

**Offline Detection**


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