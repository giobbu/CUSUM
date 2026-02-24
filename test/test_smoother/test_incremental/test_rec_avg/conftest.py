import pytest
from source.smoother.incremental import RecursiveAverage

@pytest.fixture
def smoother():
    """Fixture to initialize RecursiveAverage instance."""
    return RecursiveAverage()