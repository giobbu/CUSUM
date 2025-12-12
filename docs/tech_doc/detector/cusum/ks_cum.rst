

KS-CUM Detector
---------------

.. autoclass:: source.detector.cusum.KS_CUM_Detector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Example 
-------

**Offline Detection**

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