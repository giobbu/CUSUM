import pytest
import numpy as np

def test_fit_with_invalid_input(smoother):
    """Test fit method with invalid input."""
    with pytest.raises(ValueError):
        smoother.fit("invalid_input")  # Not a list
    with pytest.raises(ValueError):
        smoother.fit(np.array([1.0, 2.0]))  # Not a list of numpy arrays