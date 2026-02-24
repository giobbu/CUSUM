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

def test_attr_initialization():
    """Test initialization of attributes."""
    detector = CUSUM_Detector(warmup_period=10, delta=10, threshold=20)
    # assert existance of attributes and their initial values
    assert hasattr(detector, 'warmup_period')
    assert hasattr(detector, 'delta')
    assert hasattr(detector, 'threshold')

def test_methods_exist():
    """Test existence of methods."""
    detector = CUSUM_Detector(warmup_period=10, delta=10, threshold=20)
    assert hasattr(detector, 'detection')
    assert hasattr(detector, 'offline_detection')


