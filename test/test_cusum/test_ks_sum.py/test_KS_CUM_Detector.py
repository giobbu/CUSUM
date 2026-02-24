import numpy as np
import pytest




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
    expected_keys = {"ks_statistics", "p_values", "is_drift", "change_points"}
    assert set(results.keys()) == expected_keys

def test_offline_detection_results_length(detector):
    """Test offline_detection method returns results of correct length."""
    data = np.random.normal(0, 1, 100)
    results = detector.offline_detection(data)
    assert len(results["ks_statistics"]) == len(data)
    assert len(results["p_values"]) == len(data)
    assert len(results["is_drift"]) == len(data)