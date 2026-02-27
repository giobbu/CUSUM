import pytest
import numpy as np

def test_update_with_invalid_input(smoother):
    """Test update method with invalid input."""
    with pytest.raises(ValueError):
        smoother.update("invalid_input")  # Not a numpy array
    with pytest.raises(ValueError):
        smoother.update(np.array([[1, 2], [3, 4]]))  # Not a 1D array

def test_update_with_valid_input(smoother):
    """Test update method with valid input."""
    observation = np.array([1.0])
    smoother.update(observation)
    assert np.array_equal(smoother.lowpass_mean, observation)
    assert smoother.num_iterations == 1

def test_update_with_multiple_observations(smoother):
    """Test update method with multiple observations."""
    observations = [np.array([1.0]), np.array([2.0])]
    for obs in observations:
        smoother.update(obs)
    expected_lowpass_mean = smoother.alpha*observations[0] + (1-smoother.alpha)*observations[1]
    assert np.array_equal(smoother.lowpass_mean, expected_lowpass_mean)
    assert smoother.num_iterations == len(observations)
