import numpy as np
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

def test_detect_change_points_with_invalid_data_type(detector):
    """Test detect_change_points method with invalid data type."""
    data = [12.3, 14.5, 15.6, 16.8, 17.9]
    with pytest.raises(ValueError):
        detector.offline_detection(data)

def test_detection_before_warmup_period(detector):
    """Test detection method before reaching the warmup period."""
    for i in range(1, detector.warmup_period):
        upper, lower, cusum, is_changepoint = detector.detection(i)
        assert upper == 0
        assert lower == 0
        assert cusum == 0
        assert not is_changepoint

def test_detection_at_warmup_period(detector):
    """Test detection method at the warmup period."""
    observation = detector.warmup_period
    upper, lower, cusum, is_changepoint = detector.detection(observation)
    assert upper == 0
    assert lower == 0
    assert cusum == 0
    assert not is_changepoint
    assert detector.current_obs[0] == detector.warmup_period


def test_detection_after_warmup_period_without_changepoint(detector):
    """Test detection method after the warmup period without a changepoint."""
    observations = range(detector.warmup_period + 1, detector.warmup_period + 6)
    results = [detector.detection(observation) for observation in observations]
    upper_limits, lower_limits, cusums, is_changepoints = zip(*results)
    assert all(upper == 0 for upper in upper_limits)
    assert all(lower == 0 for lower in lower_limits)
    assert all(cusum == 0 for cusum in cusums)
    assert all(not is_changepoint for is_changepoint in is_changepoints)

def test_detection_after_warmup_period_with_changepoint(detector):
    """Test detection method after the warmup period without and with a changepoint."""

    detector.warmup_period = 50

    #Predict next observations without any changepoint
    observations = np.arange(1, detector.warmup_period+1, 1)
    results = [detector.detection(observation) for observation in observations]
    upper_limits, lower_limits, cusums, is_changepoints = zip(*results)
    # Assertion for no changepoint
    assert all((cusum >= lower or cusum <= upper) for cusum, lower, upper in zip(cusums, lower_limits, upper_limits))
    assert all(not is_changepoint for is_changepoint in is_changepoints)

    # Predict next observations with a changepoint
    # Generate array after drift
    observations = np.random.normal(1, 1, 1000)
    drift_observations = np.random.normal(1000, 1, 1000)
    new_observations = np.concatenate((observations, drift_observations))
    results = [detector.detection(observation) for observation in new_observations]
    upper_limits, lower_limits, cusums, is_changepoints = zip(*results)
    # Assertion for a changepoint
    assert any(is_cp for is_cp in is_changepoints)