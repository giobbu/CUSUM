from source.model.incremental import StochasticGradientDescent
import numpy as np
import pytest

def test_attr_initialization():
    """Test initialization of attributes."""
    model = StochasticGradientDescent(num_variables=3, learning_rate=0.001)
    # assert existance of attributes and their initial values
    assert hasattr(model, 'num_variables')
    assert hasattr(model, 'learning_rate')
    assert hasattr(model, 'w')
    assert hasattr(model, 'num_observations')
    assert hasattr(model, 'residual')
    assert hasattr(model, 'residual_sqr')
    assert hasattr(model, 'rmse')


def test_init_with_invalid_n_features():
    """Test initialization with invalid n_features."""
    with pytest.raises(ValueError):
        StochasticGradientDescent(num_variables=0, learning_rate=0.001)

def test_init_with_invalid_forgetting_factor_negative():
    """Test initialization with invalid forgetting factor."""
    with pytest.raises(ValueError):
        StochasticGradientDescent(num_variables=3, learning_rate=-0.001)

def test_init_with_invalid_forgetting_factor_greater_than_one():
    """Test initialization with forgetting factor greater than one."""
    with pytest.raises(ValueError):
        StochasticGradientDescent(num_variables=3, learning_rate=1.5)


def test_attr_types():
    """Test types of attributes."""
    model = StochasticGradientDescent(num_variables=3, learning_rate=0.001)
    assert isinstance(model.num_variables, int)
    assert isinstance(model.learning_rate, float)
    assert isinstance(model.w, np.ndarray)
    assert isinstance(model.num_observations, int)
    assert isinstance(model.residual, np.ndarray)
    assert isinstance(model.residual_sqr, np.ndarray)
    assert isinstance(model.rmse, np.ndarray)

def test_methods_exist_update():
    """Test existence of methods."""
    model = StochasticGradientDescent(num_variables=3, learning_rate=0.001)
    assert hasattr(model, 'update')

def test_methods_exist_predict():
    """Test existence of methods."""
    model = StochasticGradientDescent(num_variables=3, learning_rate=0.001)
    assert hasattr(model, 'predict')

def test_methods_exist_fit():
    """Test existence of methods."""
    model = StochasticGradientDescent(num_variables=3, learning_rate=0.001)
    assert hasattr(model, 'fit')



