from source.model.naive import Persistent

def test_attr_initialization():
    """Test initialization of attributes."""
    model = Persistent()
    # assert existance of attributes and their initial values
    assert hasattr(model, 'num_variables')
    assert hasattr(model, 'num_observations')

def test_attr_types():
    """Test types of attributes."""
    model = Persistent()
    assert isinstance(model.num_variables, int)
    assert isinstance(model.num_observations, int)

def test_methods_exist_update():
    """Test existence of methods."""
    model = Persistent()
    assert hasattr(model, 'update')

def test_methods_exist_predict():
    """Test existence of methods."""
    model = Persistent()
    assert hasattr(model, 'predict')

def test_methods_exist_fit():
    """Test existence of methods."""
    model = Persistent()
    assert hasattr(model, 'fit')