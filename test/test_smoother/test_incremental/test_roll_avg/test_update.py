import pytest
import numpy as np

def test_update_with_invalid_input(smoother):
    """Test update method with invalid input."""
    with pytest.raises(ValueError):
        smoother.update("invalid_input")  # Not a numpy array
    with pytest.raises(ValueError):
        smoother.update(np.array([[1, 2], [3, 4]]))  # Not a 1D array


def test_update_with_observations_equal_to_window(smoother):
    """Test update method with multiple observations equal to window size."""
    observations = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
    num_iterations = 0
    for obs in observations:
        num_iterations += 1
        if num_iterations == 1:
            observe_0 = obs
            smoother.update(obs)
            mean_1 = smoother.moving_mean
            assert np.array_equal(smoother.moving_mean, obs)
        elif num_iterations < smoother.window:
            alpha = (num_iterations-1)/num_iterations
            expected_moving_mean = alpha*mean_1 + (1-alpha)*obs
            smoother.update(obs)
            assert np.array_equal(smoother.moving_mean, expected_moving_mean)
            mean_2 = smoother.moving_mean
        else:
            expected_moving_mean = mean_2 + (obs - observe_0)/smoother.window
            smoother.update(obs)
            assert np.array_equal(smoother.moving_mean, expected_moving_mean)
        
