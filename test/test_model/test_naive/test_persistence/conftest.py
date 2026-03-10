from source.model.naive import Persistent
import pytest


@pytest.fixture
def model():
    """Fixture to initialize Persistent instance."""
    return Persistent()