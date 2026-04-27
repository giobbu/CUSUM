import numpy as np
import pytest

# --- update store ---

    # def update_store(self, X_new, y_new):
    #     """
    #     Update the data store with new observations and labels.

    #     Parameters
    #     ----------
    #     X_new : numpy array
    #         New observation vectors.
    #     y_new : numpy array
    #         New labels corresponding to the observations.
    #     """
    #     if self.X is None:
    #         self.X = X_new
    #         self.y = y_new
    #     elif len(self.X) >= self.memory_size:
    #         self.X = np.vstack((self.X, X_new))
    #         self.y = np.hstack((self.y, y_new))
    #         self.X = self.X[-self.memory_size:]
    #         self.y = self.y[-self.memory_size:]
    #     else:
    #         self.X = np.vstack((self.X, X_new))
    #         self.y = np.hstack((self.y, y_new))

def test_update_store_initialization(model):
    """Test that the store is initialized correctly when it is empty."""
    X_new = np.array([[1, 2], [3, 4]])
    y_new = np.array([0, 1])

    model.update_store(X_new, y_new)
    
    assert np.array_equal(model.X, X_new)
    assert np.array_equal(model.y, y_new)

def test_update_store_with_existing_data(model):
    """Test that new data is added correctly when the store already has data."""
    model.X = np.array([[1, 2], [3, 4]])
    model.y = np.array([0, 1])

    X_new = np.array([[5, 6], [7, 8]])
    y_new = np.array([0, 1])
    model.update_store(X_new, y_new)

    expected_X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    expected_y = np.array([0, 1, 0, 1])

    assert np.array_equal(model.X, expected_X)
    assert np.array_equal(model.y, expected_y)
