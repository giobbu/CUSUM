import pytest
from source.generator.change_point_generator import ChangePointGenerator

def test_init_with_invalid_num_segments_zero():
    """Test initialization with invalid num_segments."""
    with pytest.raises(ValueError):
        ChangePointGenerator(num_segments=0, segment_length=500, change_point_type='sudden_shift', seed=42)

def test_init_with_invalid_num_segments_int():
    """Test initialization with invalid num_segments."""
    with pytest.raises(ValueError):
        ChangePointGenerator(num_segments=1.5, segment_length=500, change_point_type='sudden_shift', seed=42)

def test_init_with_invalid_segment_length_zero():
    """Test initialization with invalid segment_length."""
    with pytest.raises(ValueError):
        ChangePointGenerator(num_segments=3, segment_length=0, change_point_type='sudden_shift', seed=42)

def test_init_with_invalid_segment_length_int():
    """Test initialization with invalid segment_length."""
    with pytest.raises(ValueError):
        ChangePointGenerator(num_segments=3, segment_length=5.3, change_point_type='sudden_shift', seed=42)

def test_init_with_invalid_change_point_type():
    """Test initialization with invalid change_point_type."""
    with pytest.raises(ValueError):
        ChangePointGenerator(num_segments=3, segment_length=500, change_point_type='invalid_type', seed=42)

def test_init_with_invalid_seed_str():
    """Test initialization with invalid seed."""
    with pytest.raises(ValueError):
        ChangePointGenerator(num_segments=3, segment_length=500, change_point_type='sudden_shift', seed='not_an_int')

