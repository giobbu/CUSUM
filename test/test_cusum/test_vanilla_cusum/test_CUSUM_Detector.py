import pytest
import numpy as np


def test_detection_before_warmup_period(detector):
    """Test detection method before reaching the warmup period."""
    for i in range(1, detector.warmup_period):
        pos_change, neg_change, is_changepoint = detector.detection(i)
        assert pos_change <= detector.threshold
        assert neg_change <= detector.threshold
        assert not is_changepoint

def test_detection_at_warmup_period(detector):
    """Test detection method at the warmup period."""
    observation = detector.warmup_period
    pos_change, neg_change, is_changepoint = detector.detection(observation)
    assert pos_change <= detector.threshold
    assert neg_change <= detector.threshold
    assert not is_changepoint
    assert detector.current_obs[0] == detector.warmup_period

def test_detection_after_warmup_period_without_changepoint(detector):
    """Test detection method after the warmup period without a changepoint."""
    observations = range(detector.warmup_period + 1, detector.warmup_period + 6)
    results = [detector.detection(observation) for observation in observations]
    pos_changes, neg_changes, is_changepoints = zip(*results)
    assert all(pos_change <= detector.threshold for pos_change in pos_changes)
    assert all(neg_change <= detector.threshold for neg_change in neg_changes)
    assert all(not is_changepoint for is_changepoint in is_changepoints)

def test_detection_after_warmup_period_with_changepoint(detector):
    """Test detection method after the warmup period without and with a changepoint."""
    detector.warmup_period = 50
    # Predict next observations without any changepoint
    observations = np.arange(1, detector.warmup_period+1, 1)
    results = [detector.detection(observation) for observation in observations]
    pos_changes, neg_changes, is_changepoints = zip(*results)
    assert all(pos_change <= detector.threshold for pos_change in pos_changes)
    assert all(neg_change <= detector.threshold for neg_change in neg_changes)
    assert all(not is_changepoint for is_changepoint in is_changepoints)

    # Predict next observations with a changepoint
    # Generate array after drift
    observations = np.random.normal(1, 1, 1000)
    drift_observations = np.random.normal(1000, 1, 1000)
    new_observations = np.concatenate((observations, drift_observations))
    results = [detector.detection(observation) for observation in new_observations]
    pos_changes, neg_changes, is_changepoints = zip(*results)
    assert any(pos_change > detector.threshold for pos_change in pos_changes)
    assert any((neg_change > detector.threshold or pos_change > detector.threshold) for neg_change, pos_change in zip(neg_changes, pos_changes))
    assert any(is_changepoint for is_changepoint in is_changepoints)

def test_offline_detection_with_invalid_data_type(detector):
    """Test offline_detection method with invalid data type."""
    data = [12.3, 14.5, 15.6, 16.8, 17.9]
    with pytest.raises(ValueError):
        detector.offline_detection(data)

def test_offline_detection_with_invalid_warmup_length(detector):
    """Test offline_detection method with invalid warmup length."""
    data = np.random.normal(0, 1, 100)
    detector.warmup_period = 150
    with pytest.raises(ValueError):
        detector.offline_detection(data)

def test_offline_detection_results_keys(detector):
    """Test offline_detection method returns results with expected keys."""
    data = np.random.normal(0, 1, 100)
    results = detector.offline_detection(data)
    expected_keys = {"pos_changes", "neg_changes", "is_drift", "change_points"}
    assert set(results.keys()) == expected_keys

def test_offline_detection_results_length(detector):
    """Test offline_detection method returns results of correct length."""
    data = np.random.normal(0, 1, 100)
    results = detector.offline_detection(data)
    assert len(results["pos_changes"]) == len(data)
    assert len(results["neg_changes"]) == len(data)
    assert len(results["is_drift"]) == len(data)