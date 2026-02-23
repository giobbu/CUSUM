Page-Hinkley Test Detector Class
----------------------------

.. autoclass:: source.detector.cusum.PHTest_Detector
   :members: 
   :undoc-members:  
   :show-inheritance:
   :special-members: __init__

Examples
--------

**Instance-based Detection**

.. code-block:: python

    from source.detector.cusum import PHTest_Detector

    ph_test_detector = PHTest_Detector(warmup_period=10, delta=10, threshold=20)
    data_stream = np.concatenate([np.random.normal(0, 1, 100),
                        np.random.normal(5, 1, 100)])
    for data in data_stream:
        pos, neg, is_change = ph_test_detector.detection(data)
        print(f"Change Detected: {is_change} \n -Positives: {pos[0]}, \n -Negatives: {neg[0]}")


**Batch-based Detection**

.. code-block:: python

    from source.detector.cusum import PHTest_Detector

    ph_test_detector = PHTest_Detector(warmup_period=10, delta=10, threshold=20)
    data = np.concatenate([np.random.normal(0, 1, 100),
                        np.random.normal(5, 1, 100)])
    results = ph_test_detector.offline_detection(data)
    ph_test_detector.plot_change_points(data,
                                results["change_points"],
                                results["pos_changes"],
                                results["neg_changes"])