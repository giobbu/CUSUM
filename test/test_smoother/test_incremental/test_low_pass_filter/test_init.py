import pytest
from source.smoother.incremental import LowPassFilter

def test_negative_alpha_raises_value_error(smoother):
    """Test that initializing LowPassFilter with negative alpha raises ValueError."""
    with pytest.raises(ValueError):
        LowPassFilter(alpha=-0.5)

def test_alpha_greater_than_one_raises_value_error(smoother):
    """Test that initializing LowPassFilter with alpha greater than 1 raises ValueError."""
    with pytest.raises(ValueError):
        LowPassFilter(alpha=1.5)

def test_recursive_average_initialization():
    """Test the initialization of RecursiveAverage."""
    smoother = LowPassFilter()
    assert smoother.alpha == 0.999
    assert smoother.num_iterations == 0
    assert smoother.lowpass_mean is None

def test_recursive_average_methods_exist():
    """Test the existence of methods in RecursiveAverage."""
    smoother = LowPassFilter()
    assert hasattr(smoother, 'update')
    assert hasattr(smoother, 'fit')

def test_default_values_after_initialization(smoother):
    """Test default values after initialization."""
    assert smoother.alpha == 0.999
    assert smoother.num_iterations == 0
    assert smoother.lowpass_mean is None