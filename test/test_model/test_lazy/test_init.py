
import numpy as np

# class WeightedKNN:
#     """
#     Weighted K-Nearest Neighbors with exponential decay weighting.

#     Parameters
#     ----------
#     alpha : float
#         Decay rate for the weights.
#     k : int
#         Number of nearest neighbors to consider.
#     decay : str
#         Type of decay to use ("exponential", "linear" supported).
#     bandwidth : float
#         Bandwidth for the KDE.
#     """

#     def __init__(self, alpha, k=5, decay="exponential", bandwidth = 1.0):
#         """
#         Initialize the WeightedKNN object.
        
#         Parameters
#         ----------
#         alpha : float
#             Decay rate for the weights.
#         k : int
#             Number of nearest neighbors to consider.
#         decay : str
#             Type of decay to use ("exponential", "linear" supported).
#         bandwidth : float
#             Bandwidth for the KDE.
#         """
#         if len(k) <= 0:
#             raise ValueError("k must be a positive integer.")
#         if decay not in ["exponential", "linear"]:
#             raise ValueError("Unsupported decay type. Use 'exponential' or 'linear'.")
#         if alpha <= 0:
#             raise ValueError("Alpha must be a positive number.")
#         if bandwidth <= 0:
#             raise ValueError("Bandwidth must be a positive number.")
        
#         self.alpha = alpha
#         self.k = k
#         self.decay = decay
#         self.bandwidth = bandwidth

from source.model.lazy import WeightedKNN
import numpy as np
import pytest

def test_init_with_k_zero_value():
    """Test initialization with invalid k value."""
    with pytest.raises(ValueError):
        WeightedKNN(alpha=0.5, k=0, decay="exponential", bandwidth=1.0)

def test_init_with_k_negative_value():
    """Test initialization with invalid k value."""
    with pytest.raises(ValueError):
        WeightedKNN(alpha=0.5, k=-1, decay="exponential", bandwidth=1.0)

def test_init_with_invalid_decay():
    """Test initialization with invalid decay type."""
    with pytest.raises(ValueError):
        WeightedKNN(alpha=0.5, k=5, decay="invalid_decay", bandwidth=1.0)

def test_init_with_alpha_zero_value():
    """Test initialization with zero alpha value."""
    with pytest.raises(ValueError):
        WeightedKNN(alpha=0, k=5, decay="exponential", bandwidth=1.0)

def test_init_with_alpha_negative_value():
    """Test initialization with negative alpha value."""
    with pytest.raises(ValueError):
        WeightedKNN(alpha=-0.5, k=5, decay="exponential", bandwidth=1.0)

def test_init_with_bandwidth_zero_value():
    """Test initialization with zero bandwidth value."""
    with pytest.raises(ValueError):
        WeightedKNN(alpha=0.5, k=5, decay="exponential", bandwidth=0)

def test_attributes_assigned_correctly():
    obj = WeightedKNN(alpha=0.3, k=3, decay="linear", bandwidth=2.0)
    assert obj.alpha == 0.3
    assert obj.k == 3
    assert obj.decay == "linear"
    assert obj.bandwidth == 2.0

def test_default_values():
    obj = WeightedKNN(alpha=0.5)
    assert obj.k == 5
    assert obj.decay == "exponential"
    assert obj.bandwidth == 1.0