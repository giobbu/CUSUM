import pytest
import numpy as np

def test_update_method_observation_none(model):
    """Test update method with observation being None."""
    with pytest.raises(ValueError):
        model.update(None)

def test_update_method_observation_dimension_mismatch(model):
    """Test update method with observation dimension mismatch."""
    # two observations instead of one
    observation = np.array([[1], [2]])  # 2 observations instead of 1
    with pytest.raises(ValueError):
        model.update(observation)

def test_update_method_observation_not_column_vector(model):
    """Test update method with observation not being a column vector."""
    # 2 variables instead of 1
    observation = np.array([[1, 2]])  # 2 variables instead of
    with pytest.raises(ValueError):
        model.update(observation)

def test_update_method_valid_observation(model):
    """Test update method with a valid observation."""
    observation = np.array([[1]])  # Valid observation
    assert model.last_observation == np.zeros((1, model.num_variables))
    model.update(observation)
    assert model.num_observations == 1
    assert np.array_equal(model.last_observation, observation)

def test_fit_method_observations_none(model):
    """Test fit method with observations being None."""
    with pytest.raises(ValueError):
        model.fit(None)

def test_fit_method_observations_not_list(model):
    """Test fit method with observations not being a list."""
    with pytest.raises(ValueError):
        model.fit(np.array([[1], [2], [3]]))  # Not a list

def test_fit_method_observations_empty_list(model):
    """Test fit method with observations being an empty list."""
    with pytest.raises(ValueError):
        model.fit([])

def test_fit_method_valid_observations(model):
    """Test fit method with valid observations."""
    observations = [np.array([[1]]), np.array([[2]]), np.array([[3]])]  # Valid observations
    model.fit(observations)
    assert model.num_observations == 3
    assert np.array_equal(model.last_observation, observations[-1])

def predict_method_valid_observation(model):
    """Test predict method with a valid observation."""
    observation = np.array([[1]])  # Valid observation
    model.update(observation)  # Update model with a valid observation
    prediction = model.predict()
    assert isinstance(prediction, float)
    assert not np.isnan(prediction)
    assert prediction == observation[0, 0]  # Prediction should match the last observation
