import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from source.detector.cusum import CUSUM_Detector
import pytest
import numpy as np


@pytest.fixture
def detector():
    """Fixture to initialize CUSUM_Detector instance with default parameters."""
    return CUSUM_Detector(warmup_period=10, delta=10, threshold=20)


def test_init(detector):
    # Test with custom parameters
    assert detector.warmup_period == 10
    assert detector.delta == 10
    assert detector.threshold == 20


def test_predict_next(detector):
    observation = 25.7
    pos_changes, neg_changes, is_changepoint = detector.predict_next(observation)
    assert np.isclose(pos_changes, 0, atol=0.001)
    assert np.isclose(neg_changes, 0, atol=0.001)
    assert not is_changepoint


def test_init_params_with_valid_data(detector):
    """Test _init_params method with valid data."""
    detector.current_obs = [20, 22, 24, 26, 28]
    detector._init_params()
    assert not np.isnan(detector.current_mean)
    assert not np.isnan(detector.current_std)
    assert detector.z == 0
    assert detector.S_pos == 0
    assert detector.S_neg == 0

def test_compute_cumusum(detector):
    detector.current_obs = [20, 22, 24, 26, 28]
    detector._init_params()
    detector._compute_cumusum()
    assert np.isclose(detector.S_pos, 0, atol=0.001)
    assert np.isclose(detector.S_neg, 0, atol=0.001)

def test_predict_next(detector):
    observation = 25.7
    pos_changes, neg_changes, is_changepoint = detector.predict_next(observation)
    assert np.isclose(pos_changes, 0, atol=0.001)
    assert np.isclose(neg_changes, 0, atol=0.001)
    assert not is_changepoint

def test_detect_change_points_with_invalid_data_length(detector):
    """Test detect_change_points method with invalid data length."""
    data = np.array([12.3, 14.5, 15.6])
    with pytest.raises(ValueError):
        detector.detect_change_points(data)

def test_detect_change_points_with_invalid_data_type(detector):
    """Test detect_change_points method with invalid data type."""
    data = [12.3, 14.5, 15.6, 16.8, 17.9]
    with pytest.raises(ValueError):
        detector.detect_change_points(data)