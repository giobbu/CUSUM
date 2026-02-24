from source.model.incremental import RecursiveLeastSquares
import pytest


@pytest.fixture
def model():
    """Fixture to initialize RecursiveLeastSquares instance."""
    return RecursiveLeastSquares(num_variables=3, forgetting_factor=0.99, initial_delta=1.0)