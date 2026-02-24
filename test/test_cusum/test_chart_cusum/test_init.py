import pytest
from source.detector.cusum import ChartCUSUM_Detector

def test_init_with_invalid_warmup_period():
    """Test initialization with invalid warmup_period."""
    with pytest.raises(ValueError):
        ChartCUSUM_Detector(warmup_period=5, level=3, deviation_type='sqr-dev')

def test_init_with_invalid_warmup_period():
    """Test initialization with invalid level."""
    with pytest.raises(ValueError):
        ChartCUSUM_Detector(warmup_period=10, level=10, deviation_type='sqr-dev')

def test_init_with_invalid_deviation_type():
    """Test initialization with invalid deviation_type."""
    with pytest.raises(ValueError):
        ChartCUSUM_Detector(warmup_period=10, level=2, deviation_type='invalid')

def test_attr_initialization():
    """Test initialization of attributes."""
    detector = ChartCUSUM_Detector(warmup_period=10, level=3, deviation_type='sqr-dev')
    # assert existance of attributes and their initial values
    assert hasattr(detector, 'warmup_period')
    assert hasattr(detector, 'level')
    assert hasattr(detector, 'deviation_type')

def test_methods_exist():
    """Test existence of methods."""
    detector = ChartCUSUM_Detector(warmup_period=10, level=3, deviation_type='sqr-dev')
    assert hasattr(detector, 'detection')
    assert hasattr(detector, 'offline_detection')