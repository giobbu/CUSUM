Majority Vote Alerting
======================

The majority vote alerting method is a simple yet effective approach to detect drifts. It works by aggregating the results of multiple detectors or the same detector over sevearl instances and making a decision based on the majority of the votes. This method can help to reduce false positives and improve the overall accuracy of change point detection.

Majorty Vote Alerting Classes
-----------------------------

.. autoclass:: source.alerting.majority_vote.MajorityVote
   :members: 
   :undoc-members:  
   :show-inheritance:
   :special-members: __init__