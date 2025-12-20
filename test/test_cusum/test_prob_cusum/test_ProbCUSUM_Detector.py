import numpy as np
import pytest

def test_detect_change_points_with_invalid_data_type(detector):
    """Test detect_change_points method with invalid data type."""
    data = [12.3, 14.5, 15.6, 16.8, 17.9]
    with pytest.raises(ValueError):
        detector.offline_detection(data)

def test_detection_before_warmup_period(detector):
    """Test detection method before reaching the warmup period."""
    for i in range(1, detector.warmup_period):
        pos_change, neg_change, is_changepoint = detector.detection(i)
        assert pos_change <= detector.threshold
        assert neg_change <= detector.threshold
        assert not is_changepoint

def test_detection_before_warmup_period(detector):
    """Test detection method before reaching the warmup period."""
    for i in range(1, detector.warmup_period):
        probability, is_changepoint = detector.detection(i)
        assert (1 - probability) >= detector.threshold_probability  # p-value greater than critical value (can not reject the null hypothesis)
        assert not is_changepoint

def test_detection_at_warmup_period(detector):
    """Test detection method at the warmup period."""
    observation = detector.warmup_period
    probability, is_changepoint = detector.detection(observation)
    assert (1 - probability) >= detector.threshold_probability 
    assert not is_changepoint
    assert detector.observations[0] == detector.warmup_period


def test_detection_after_warmup_period_without_changepoint(detector):
    """Test detection method after the warmup period without a changepoint."""
    observations = range(detector.warmup_period + 1, detector.warmup_period + 6)
    results = [detector.detection(observation) for observation in observations]
    probabilities, is_changepoints = zip(*results)
    assert all((1 - probability) >= detector.threshold_probability  for probability in probabilities)
    assert all(not is_changepoint for is_changepoint in is_changepoints)


def test_detection_after_warmup_period_with_changepoint(detector):
    """Test detection method after the warmup period without and with a changepoint."""
    detector.warmup_period = 50
    # Predict next observations without any changepoint
    observations = np.arange(1, detector.warmup_period+1, 1)
    results = [detector.detection(observation) for observation in observations]
    probabilities, is_changepoints = zip(*results)
    assert all((1 - probability) >= detector.threshold_probability  for probability in probabilities)
    assert all(not is_changepoint for is_changepoint in is_changepoints)

    # Predict next observations with a changepoint
    # Generate array after drift
    observations = np.random.normal(1, 1, 1000)
    drift_observations = np.random.normal(1000, 1, 1000)
    new_observations = np.concatenate((observations, drift_observations))
    results = [detector.detection(observation) for observation in new_observations]
    probabilities, is_changepoints = zip(*results)
    assert any((1 - probability) < detector.threshold_probability  for probability in probabilities)
    assert any(not is_changepoint for is_changepoint in is_changepoints)