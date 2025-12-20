from source.detector.cusum import CUSUM_Detector
import pytest

def test_init_with_invalid_warmup_period():
    """Test initialization with invalid warmup_period."""
    with pytest.raises(ValueError):
        CUSUM_Detector(warmup_period=5, delta=10, threshold=20)

def test_init_with_invalid_warmup_period():
    """Test initialization with invalid delta."""
    with pytest.raises(ValueError):
        CUSUM_Detector(warmup_period=5, delta=1, threshold=20)

def test_init_with_invalid_deviation_type():
    """Test initialization with invalid threshold."""
    with pytest.raises(ValueError):
        CUSUM_Detector(warmup_period=5, delta=10, threshold=2)