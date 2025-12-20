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