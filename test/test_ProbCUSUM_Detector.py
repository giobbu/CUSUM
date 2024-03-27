import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from source.detector.cusum import ProbCUSUM_Detector
import pytest
import numpy as np

@pytest.fixture
def detector():
    """Fixture to initialize CUSUM_Detector instance with default parameters."""
    return ProbCUSUM_Detector(warmup_period=10, threshold_probability=0.01)

def test_init_with_invalid_warmup_period():
    """Test initialization with invalid warmup_period."""
    with pytest.raises(ValueError):
        ProbCUSUM_Detector(warmup_period=5, threshold_probability=0.1)

def test_init_with_invalid_threshold_probability():
    """Test initialization with invalid threshold_probability."""
    with pytest.raises(ValueError):
        ProbCUSUM_Detector(warmup_period=10, threshold_probability=1.1)

def test_predict_next_before_warmup_period(detector):
    """Test predict_next method before reaching the warmup period."""
    for i in range(1, detector.warmup_period):
        probability, is_changepoint = detector.predict_next(i)
        assert probability == 0
        assert not is_changepoint

def test_predict_next_at_warmup_period(detector):
    """Test predict_next method at the warmup period."""
    observation = detector.warmup_period
    probability, is_changepoint = detector.predict_next(observation)
    assert probability == 0
    assert not is_changepoint
    assert detector.observations[0] == detector.warmup_period

def test_predict_next_after_warmup_period_without_changepoint(detector):
    """Test predict_next method after the warmup period without a changepoint."""
    #Predict next observations without any change point
    observations = range(1, detector.warmup_period + 1)
    for observation in observations:
        _, is_changepoint = detector.predict_next(observation)
        assert not is_changepoint
    changepoint = 1
    probability, is_changepoint = detector.predict_next(detector.warmup_period + changepoint)
    assert (1-probability) > detector.threshold_probability
    assert not is_changepoint


def test_predict_next_after_warmup_period_with_changepoint(detector):
    """Test predict_next method after the warmup period with a changepoint."""
    observations = range(1, detector.warmup_period + 1)
    for observation in observations:
        _, is_changepoint = detector.predict_next(observation)
        assert not is_changepoint
    changepoint = 100
    probability, is_changepoint = detector.predict_next(detector.warmup_period + changepoint)
    assert (1-probability) < detector.threshold_probability
    assert is_changepoint