from source.model.incremental import StochasticGradientDescent
import pytest


@pytest.fixture
def model():
    """Fixture to initialize RecursiveLeastSquares instance."""
    return StochasticGradientDescent(num_variables=3, learning_rate=0.001)

