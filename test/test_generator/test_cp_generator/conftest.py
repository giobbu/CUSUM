from source.generator.change_point_generator import ChangePointGenerator
import pytest


@pytest.fixture
def generator():
    """Fixture to initialize ChangePointGenerator instance."""
    return ChangePointGenerator(num_segments=3, segment_length=500, change_point_type='sudden_shift', seed=42)
