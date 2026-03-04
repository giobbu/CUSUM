import pytest
import numpy as np

def test_update_method_observation_dimension_mismatch(model):
    """Test update method with observation dimension mismatch."""
    observation = np.array([[1], [2]])  # 2 variables instead of 3
    label = 5.0
    with pytest.raises(ValueError):
        model.update(observation, label)

def test_update_method_observation_not_column_vector(model):
    """Test update method with observation not being a column vector."""
    observation = np.array([1, 2, 3])  # Not a column vector
    label = 5.0
    with pytest.raises(ValueError):
        model.update(observation, label)

def test_update_method_label_not_single_value(model):
    """Test update method with label not being a single value."""
    observation = np.array([[1], [2], [3]])
    label = [5.0, 6.0]  # Not a single value
    with pytest.raises(ValueError):
        model.update(observation, label)

def test_update_method_label_not_float(model):
    """Test update method with label not being a float."""
    observation = np.array([[1], [2], [3]])
    label = "5.0"  # Not a float
    with pytest.raises(ValueError):
        model.update(observation, label)
    
def test_predict_method_observation_dimension_mismatch(model):
    """Test predict method with observation dimension mismatch."""
    observation = np.array([[1], [2]])  # 2 variables instead of 3
    with pytest.raises(ValueError):
        model.predict(observation)

def test_predict_method_valid_observation(model):
    """Test predict method with a valid observation."""
    observation = np.array([[1], [2], [3]])
    prediction = model.predict(observation)
    assert isinstance(prediction, float)
    assert not np.isnan(prediction)


