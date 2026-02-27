import pytest
from source.smoother.incremental import RollingAverageFilter

@pytest.fixture
def smoother():
    """Fixture for creating a RollingAverageFilter instance."""
    return RollingAverageFilter()