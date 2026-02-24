from source.detector.cusum import ProbCUSUM_Detector
import pytest

def test_init_with_invalid_warmup_period():
    """Test initialization with invalid warmup_period."""
    with pytest.raises(ValueError):
        ProbCUSUM_Detector(warmup_period=5, threshold_probability=0.1)

def test_init_with_invalid_threshold_probability():
    """Test initialization with invalid threshold_probability."""
    with pytest.raises(ValueError):
        ProbCUSUM_Detector(warmup_period=10, threshold_probability=1.1)

def test_attr_initialization():
    """Test initialization of attributes."""
    detector = ProbCUSUM_Detector(warmup_period=10, threshold_probability=0.99)
    # assert existance of attributes and their initial values
    assert hasattr(detector, 'warmup_period')
    assert hasattr(detector, 'threshold_probability')

def test_methods_exist():
    """Test existence of methods."""
    detector = ProbCUSUM_Detector(warmup_period=10, threshold_probability=0.99)
    assert hasattr(detector, 'detection')
    assert hasattr(detector, 'offline_detection')