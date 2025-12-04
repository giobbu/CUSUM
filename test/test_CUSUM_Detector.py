from source.detector.cusum import CUSUM_Detector
import pytest
import numpy as np


@pytest.fixture
def detector():
    """Fixture to initialize CUSUM_Detector instance with default parameters."""
    return CUSUM_Detector(warmup_period=10, delta=10, threshold=20)

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

def test_detect_change_points_with_invalid_data_type(detector):
    """Test detect_change_points method with invalid data type."""
    data = [12.3, 14.5, 15.6, 16.8, 17.9]
    with pytest.raises(ValueError):
        detector.offline_detection(data)

def test_predict_next_before_warmup_period(detector):
    """Test predict_next method before reaching the warmup period."""
    for i in range(1, detector.warmup_period):
        pos_change, neg_change, is_changepoint = detector.predict_next(i)
        assert pos_change <= detector.threshold
        assert neg_change <= detector.threshold
        assert not is_changepoint

def test_predict_next_at_warmup_period(detector):
    """Test predict_next method at the warmup period."""
    observation = detector.warmup_period
    pos_change, neg_change, is_changepoint = detector.predict_next(observation)
    assert pos_change <= detector.threshold
    assert neg_change <= detector.threshold
    assert not is_changepoint
    assert detector.current_obs[0] == detector.warmup_period

def test_predict_next_after_warmup_period_without_changepoint(detector):
    """Test predict_next method after the warmup period without a changepoint."""
    observations = range(detector.warmup_period + 1, detector.warmup_period + 6)
    results = [detector.predict_next(observation) for observation in observations]
    pos_changes, neg_changes, is_changepoints = zip(*results)
    assert all(pos_change <= detector.threshold for pos_change in pos_changes)
    assert all(neg_change <= detector.threshold for neg_change in neg_changes)
    assert all(not is_changepoint for is_changepoint in is_changepoints)

def test_predict_next_after_warmup_period_with_changepoint(detector):
    """Test predict_next method after the warmup period without and with a changepoint."""
    detector.warmup_period = 50
    # Predict next observations without any changepoint
    observations = np.arange(1, detector.warmup_period+1, 1)
    results = [detector.predict_next(observation) for observation in observations]
    pos_changes, neg_changes, is_changepoints = zip(*results)
    assert all(pos_change <= detector.threshold for pos_change in pos_changes)
    assert all(neg_change <= detector.threshold for neg_change in neg_changes)
    assert all(not is_changepoint for is_changepoint in is_changepoints)

    # Predict next observations with a changepoint
    # Generate array after drift
    observations = np.random.normal(1, 1, 1000)
    drift_observations = np.random.normal(1000, 1, 1000)
    new_observations = np.concatenate((observations, drift_observations))
    results = [detector.predict_next(observation) for observation in new_observations]
    pos_changes, neg_changes, is_changepoints = zip(*results)
    assert any(pos_change > detector.threshold for pos_change in pos_changes)
    assert any((neg_change > detector.threshold or pos_change > detector.threshold) for neg_change, pos_change in zip(neg_changes, pos_changes))
    assert any(is_changepoint for is_changepoint in is_changepoints)