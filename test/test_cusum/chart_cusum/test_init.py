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
