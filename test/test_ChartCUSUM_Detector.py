import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from source.detector.cusum import ChartCUSUM_Detector
import pytest


@pytest.fixture
def detector():
    """Fixture to initialize ChartCUSUM_Detector instance."""
    return ChartCUSUM_Detector(warmup_period=10, level=3, deviation_type='dev')

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

def test_predict_next_before_warmup_period(detector):
    """Test predict_next method before reaching the warmup period."""
    for i in range(1, detector.warmup_period):
        upper, lower, cusum, is_changepoint = detector.predict_next(i)
        assert upper == 0
        assert lower == 0
        assert cusum == 0
        assert not is_changepoint

def test_predict_next_at_warmup_period(detector):
    """Test predict_next method at the warmup period."""

    observation = detector.warmup_period
    upper, lower, cusum, is_changepoint = detector.predict_next(observation)
    assert upper == 0
    assert lower == 0
    assert cusum == 0
    assert not is_changepoint
    assert detector.current_obs[0] == detector.warmup_period


def test_predict_next_after_warmup_period_without_changepoint(detector):
    """Test predict_next method after the warmup period without a changepoint."""
    observations = range(detector.warmup_period + 1, detector.warmup_period + 6)
    results = [detector.predict_next(observation) for observation in observations]
    upper_limits, lower_limits, cusums, is_changepoints = zip(*results)
    assert all(upper == 0 for upper in upper_limits)
    assert all(lower == 0 for lower in lower_limits)
    assert all(cusum == 0 for cusum in cusums)
    assert all(not is_changepoint for is_changepoint in is_changepoints)


def test_predict_next_after_warmup_period_without_changepoint(detector):
    """Test predict_next method after the warmup period without a changepoint."""
    #Predict next observations without any change point
    observations = range(1, detector.warmup_period + 1)
    for observation in observations:
        upper, lower, cusum, is_changepoint = detector.predict_next(observation)
        assert cusum >= lower and cusum <= upper
        assert not is_changepoint
    changepoint = 1
    upper, lower, cusum, is_changepoint = detector.predict_next(detector.warmup_period + changepoint)
    assert cusum >= lower and cusum <= upper
    assert not is_changepoint


# def test_predict_next_after_warmup_period_with_changepoint(detector):
#     """Test predict_next method after the warmup period without a changepoint."""
#     #Predict next observations without any change point
#     observations = range(1, detector.warmup_period+1)
#     for observation in observations:
#         upper, lower, cusum, is_changepoint = detector.predict_next(observation)
#         assert cusum >= lower and cusum <= upper
#         assert not is_changepoint
#     observations = [100, 2000, 0]
#     for observation in observations:
#         upper, lower, cusum, is_changepoint = detector.predict_next(observation)
#     assert cusum < lower or cusum > upper
#     assert is_changepoint