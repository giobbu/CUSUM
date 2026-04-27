from source.model.lazy import WeightedKNN
import pytest

@pytest.fixture
def model():
    """Fixture to initialize a WeightedKNN model with default parameters."""
    return WeightedKNN(alpha=0.5, k=5, decay="exponential", bandwidth=1.0, memory_size=100)