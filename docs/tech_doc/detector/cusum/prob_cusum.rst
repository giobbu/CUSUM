Probabilistic CUSUM Detector Class
----------------------------------

.. autoclass:: source.detector.cusum.ProbCUSUM_Detector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Examples
--------

**Online Detection**

.. code-block:: python

    from source.detector.cusum import ProbCUSUM_Detector

    detector = ProbCUSUM_Detector(warmup_period=10, threshold_probability=0.01)
    data_stream = np.concatenate([np.random.normal(0, 1, 100),
                        np.random.normal(5, 1, 100)])
    for data in data_stream:
        prob, is_change = detector.detection(data)
        print(f"Change Detected: {is_change} \n -Probability: {prob[0]}")

**Offline Detection**

.. code-block:: python

    from source.detector.cusum import ProbCUSUM_Detector

    detector = ProbCUSUM_Detector(warmup_period=10, threshold_probability=0.01)
    data = np.concatenate([np.random.normal(0, 1, 100),
                        np.random.normal(5, 1, 100)])
    results = detector.offline_detection(data)
    detector.plot_change_points(data,
                                results["change_points"],
                                results["probabilities"])

**Plotting**

.. image:: ../../../_static/images/probcusum.png
   :alt: CUSUM Vanilla Example
   :align: center
   :width: 600px