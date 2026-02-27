from source.smoother.incremental import RecursiveAverage

def test_recursive_average_initialization():
    """Test the initialization of RecursiveAverage."""
    smoother = RecursiveAverage()
    assert smoother.recursive_mean is None
    assert smoother.num_iterations == 0

def test_recursive_average_methods_exist():
    """Test the existence of methods in RecursiveAverage."""
    smoother = RecursiveAverage()
    assert hasattr(smoother, 'update')
    assert hasattr(smoother, 'fit')

def test_default_values_after_initialization(smoother):
    """Test default values after initialization."""
    assert smoother.recursive_mean is None
    assert smoother.num_iterations == 0