import pytest
from source.smoother.incremental import LowPassFilter

@pytest.fixture
def smoother():
    """Fixture to initialize RecursiveAverage instance."""
    return LowPassFilter()