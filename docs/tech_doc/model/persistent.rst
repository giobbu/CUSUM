Persistent Model
=================

This module contains the implementation of the persistent model for online learning. The persistent model is designed to return the last observed value as the prediction for the next time step. This simple approach can be effective in certain scenarios where the data exhibits temporal dependencies.

Persistent Model Class
----------------------
.. autoclass:: source.model.naive.Persistent
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
