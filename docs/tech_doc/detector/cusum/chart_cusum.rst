Chart CUSUM Detector
--------------------

.. autoclass:: source.detector.cusum.ChartCUSUM_Detector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Example 
-------

**Offline Detection**

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