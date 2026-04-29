import numpy as np
import pytest

def test_result_structure(model):
    """Test that the result of _apply_mean has the correct structure."""
    labels = np.array([0, 1, 1, 0])
    # Test _apply_mean
    result = model._apply_mean(labels)
    assert isinstance(result, dict)
    assert 'y_eval' in result
    assert 'kde' in result
    assert result['y_eval'] is None  # y_eval should be None for mean prediction
    # Test _apply_weighted_mean
    result = model._apply_weighted_mean(labels, np.array([0.1, 0.5, 0.3, 0.1]))
    assert isinstance(result, dict)
    assert 'y_eval' in result
    assert 'kde' in result
    assert result['y_eval'] is None  # y_eval should be None for weighted mean
    # Test _apply_kde
    model.y = np.array([0, 1, 1, 0])  # Set model.y for KDE
    result = model._apply_kde(labels)
    assert isinstance(result, dict)
    assert 'y_eval' in result
    assert 'kde' in result
    assert isinstance(result['y_eval'], np.ndarray)
    assert isinstance(result['kde'], np.ndarray)

def test_mean_prediction(model):
    """Test that the mean prediction is calculated correctly."""
    labels = np.array([0, 1, 1, 0])
    result = model._apply_mean(labels)
    expected_prediction = 0.5
    assert result['kde'] == expected_prediction

def test_weighted_mean_prediction(model):
    """Test that the weighted mean prediction is calculated correctly."""
    knn_labels = np.array([0, 1, 1, 0])
    knn_weights = np.array([0.1, 0.5, 0.3, 0.1])
    result = model._apply_weighted_mean(knn_labels, knn_weights)
    expected_prediction = (0*0.1 + 1*0.5 + 1*0.3 + 0*0.1) / (0.1 + 0.5 + 0.3 + 0.1)
    assert result['kde'] == expected_prediction

