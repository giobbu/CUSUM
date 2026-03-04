from source.model.incremental import RecursiveLeastSquares
import numpy as np
import pytest

def test_attr_initialization():
    """Test initialization of attributes."""
    model = RecursiveLeastSquares(num_variables=3, forgetting_factor=0.99, initial_delta=1.0)
    # assert existance of attributes and their initial values
    assert hasattr(model, 'num_variables')
    assert hasattr(model, 'A')
    assert hasattr(model, 'w')
    assert hasattr(model, 'forgetting_factor_inverse')
    assert hasattr(model, 'num_observations')
    assert hasattr(model, 'residual')
    assert hasattr(model, 'residual_sqr')
    assert hasattr(model, 'rmse')

    
def test_init_with_invalid_n_features():
    """Test initialization with invalid n_features."""
    with pytest.raises(ValueError):
        RecursiveLeastSquares(num_variables=0, forgetting_factor=0.99, initial_delta=1.0)

def test_init_with_invalid_forgetting_factor_negative():
    """Test initialization with invalid forgetting factor."""
    with pytest.raises(ValueError):
        RecursiveLeastSquares(num_variables=3, forgetting_factor=-0.5, initial_delta=1.0)

def test_init_with_invalid_forgetting_factor_zero():
    """Test initialization with zero forgetting factor."""
    with pytest.raises(ValueError):
        RecursiveLeastSquares(num_variables=3, forgetting_factor=0, initial_delta=1.0)

def test_init_with_invalid_initial_delta_negative():
    """Test initialization with invalid initial delta."""
    with pytest.raises(ValueError):
        RecursiveLeastSquares(num_variables=3, forgetting_factor=0.99, initial_delta=-1.0)

def test_init_with_invalid_initial_delta_zero():
    """Test initialization with zero initial delta."""
    with pytest.raises(ValueError):
        RecursiveLeastSquares(num_variables=3, forgetting_factor=0.99, initial_delta=0)



def test_attr_types():
    """Test types of attributes."""
    model = RecursiveLeastSquares(num_variables=3, forgetting_factor=0.99, initial_delta=1.0)
    assert isinstance(model.num_variables, int)
    assert isinstance(model.A, np.ndarray)
    assert isinstance(model.w, np.ndarray)
    assert isinstance(model.forgetting_factor_inverse, float)
    assert isinstance(model.num_observations, int)
    assert isinstance(model.residual, np.ndarray)
    assert isinstance(model.residual_sqr, np.ndarray)
    assert isinstance(model.rmse, np.ndarray)

def test_methods_exist_update():
    """Test existence of methods."""
    model = RecursiveLeastSquares(num_variables=3, forgetting_factor=0.99, initial_delta=1.0)
    assert hasattr(model, 'update')

def test_methods_exist_predict():
    """Test existence of methods."""
    model = RecursiveLeastSquares(num_variables=3, forgetting_factor=0.99, initial_delta=1.0)
    assert hasattr(model, 'predict')

def test_methods_exist_fit():
    """Test existence of methods."""
    model = RecursiveLeastSquares(num_variables=3, forgetting_factor=0.99, initial_delta=1.0)
    assert hasattr(model, 'fit')



