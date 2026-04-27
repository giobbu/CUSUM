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

def test_init_with_memory_size_equal_k():
    """Test initialization with memory size equal to k."""
    with pytest.raises(ValueError):
        WeightedKNN(alpha=0.5, k=5, decay="exponential", bandwidth=1.0, memory_size=5)

def test_init_with_memory_size_less_than_k():
    """Test initialization with memory size less than k."""
    with pytest.raises(ValueError):
        WeightedKNN(alpha=0.5, k=5, decay="exponential", bandwidth=1.0, memory_size=4)      

def test_attributes_assigned_correctly():
    obj = WeightedKNN(alpha=0.3, k=3, decay="linear", bandwidth=2.0, memory_size=50)
    assert obj.alpha == 0.3
    assert obj.k == 3
    assert obj.decay == "linear"
    assert obj.bandwidth == 2.0
    assert obj.memory_size == 50

def test_default_values():
    obj = WeightedKNN(alpha=0.5)
    assert obj.k == 5
    assert obj.decay == "exponential"
    assert obj.bandwidth == 1.0
    assert obj.memory_size == 100