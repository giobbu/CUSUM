import pytest
import numpy as np
from source.smoother.incremental import RollingAverageFilter

def test_negative_alpha_raises_value_error(smoother):
    """Test that initializing LowPassFilter with negative alpha raises ValueError."""
    with pytest.raises(ValueError):
        RollingAverageFilter(window=-1)

def test_recursive_average_initialization():
    """Test the initialization of RecursiveAverage."""
    smoother = RollingAverageFilter()
    assert smoother.window == 3
    assert smoother.moving_mean is None
    assert smoother.num_iterations == 0
    assert isinstance(smoother.rolling_list, list)

def test_recursive_average_methods_exist():
    """Test the existence of methods in RecursiveAverage."""
    smoother = RollingAverageFilter()
    assert hasattr(smoother, 'update')
    assert hasattr(smoother, 'fit')

def test_default_values_after_initialization(smoother):
    """Test default values after initialization."""
    assert smoother.window == 3
    assert smoother.num_iterations == 0
    assert smoother.moving_mean is None
    assert isinstance(smoother.rolling_list, list)
