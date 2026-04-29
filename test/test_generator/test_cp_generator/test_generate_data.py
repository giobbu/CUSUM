import numpy as np
import pytest

def test_add_sudden_shift(generator):
    """
    Test adding a sudden shift change point.
    """
    mean, std_dev = generator._add_sudden_shift()
    assert isinstance(mean, float)
    assert isinstance(std_dev, float)

def test_add_gradual_drift(generator):
    """
    Test adding a gradual drift change point.
    """
    mean, std_dev = generator._add_gradual_drift()
    assert isinstance(mean, np.ndarray)
    assert isinstance(std_dev, float)

def test_add_periodic_change(generator):
    """
    Test adding a periodic change point.
    """
    mean, std_dev = generator._add_periodic_change()
    assert isinstance(mean, np.ndarray)
    assert isinstance(std_dev, float)

def test_segment_data(generator):
    """
    Test the _segment_data method of ChangePointGenerator.
    """
    mean, std_dev = generator._add_sudden_shift()
    segment_data = generator._segment_data(mean, std_dev)
    assert isinstance(segment_data, np.ndarray)
    assert len(segment_data) == generator.segment_length

def test_generate_data(generator):
    """
    Test data generation of ChangePointGenerator.
    """
    generator.generate_data()
    assert len(generator.data) == generator.num_segments * generator.segment_length

def test_get_data(generator):
    """
    Test get_data method of ChangePointGenerator.
    """
    generator.generate_data()
    data = generator.get_data()
    assert isinstance(data, np.ndarray)

def test_mean_before_after_not_number(generator):
    """
    Test that add_sudden_shift raises ValueError when mean_before or mean_after is not a number.
    """
    with pytest.raises(ValueError):
        generator.add_sudden_shift(mean_before="not_a_number", mean_after=10, std_dev_before=1, std_dev_after=1, change_point_index=50)
    with pytest.raises(ValueError):
        generator.add_sudden_shift(mean_before=0, mean_after="not_a_number", std_dev_before=1, std_dev_after=1, change_point_index=50)

def test_std_dev_before_after_not_number(generator):
    """
    Test that add_sudden_shift raises ValueError when std_dev_before or std_dev_after is not a number.
    """
    with pytest.raises(ValueError):
        generator.add_sudden_shift(mean_before=0, mean_after=10, std_dev_before="not_a_number", std_dev_after=1, change_point_index=50)
    with pytest.raises(ValueError):
        generator.add_sudden_shift(mean_before=0, mean_after=10, std_dev_before=1, std_dev_after="not_a_number", change_point_index=50)

def test_change_point_index_out_of_range(generator):
    """
    Test that add_sudden_shift raises ValueError when change_point_index is out of range.
    """
    with pytest.raises(ValueError):
        generator.add_sudden_shift(mean_before=0, mean_after=10, std_dev_before=1, std_dev_after=1, change_point_index=-1)
    with pytest.raises(ValueError):
        generator.add_sudden_shift(mean_before=0, mean_after=10, std_dev_before=1, std_dev_after=1, change_point_index=len(generator.data))