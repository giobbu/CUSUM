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

def test_generate_point_nans_invalid_percentage(generator):
    """
    Test that generate_point_nans raises ValueError when percentage is invalid.
    """
    with pytest.raises(ValueError):
        generator.generate_point_nans(percentage=-0.1)
    with pytest.raises(ValueError):
        generator.generate_point_nans(percentage=1.1)

def test_generate_point_nans_valid_percentage(generator):
    """
    Test that generate_point_nans returns data with the correct percentage of NaN values.
    """
    generator.generate_data()
    percentage = 0.1
    data_with_nans = generator.generate_point_nans(percentage)
    num_nan = np.sum(np.isnan(data_with_nans))
    expected_num_nan = int(len(generator.data) * percentage)
    assert num_nan == expected_num_nan

def test_get_num_blocks_invalid_percentage(generator):
    """
    Test that _get_num_blocks raises ValueError when percentage is invalid.
    """
    with pytest.raises(ValueError):
        generator._get_num_blocks(percentage=-0.1, min_block_size=5)
    with pytest.raises(ValueError):
        generator._get_num_blocks(percentage=1.1, min_block_size=5)

def test_get_num_blocks_invalid_min_block_size(generator):
    """
    Test that _get_num_blocks raises ValueError when min_block_size is invalid.
    """
    with pytest.raises(ValueError):
        generator._get_num_blocks(percentage=0.1, min_block_size=0)
    with pytest.raises(ValueError):
        generator._get_num_blocks(percentage=0.1, min_block_size=-5)

def test_get_num_blocks_valid_inputs(generator):
    """
    Test that _get_num_blocks returns correct number of blocks for valid inputs.
    """
    generator.generate_data()
    percentage = 0.1
    min_block_size = 5
    num_elements, num_blocks = generator._get_num_blocks(percentage, min_block_size)
    expected_num_nan = int(num_elements * percentage)
    expected_num_blocks = expected_num_nan // min_block_size
    assert num_elements == len(generator.data)
    assert num_blocks == expected_num_blocks

def test_get_start_indices_invalid_num_blocks(generator):
    """
    Test that _get_start_indices_for_blocks raises ValueError when num_blocks is invalid.
    """
    with pytest.raises(ValueError):
        generator._get_start_indices_for_blocks(num_elements=100, num_blocks=0, min_block_size=5)
    with pytest.raises(ValueError):
        generator._get_start_indices_for_blocks(num_elements=100, num_blocks=-1, min_block_size=5)

def test_get_start_indices_invalid_min_block_size(generator):
    """
    Test that _get_start_indices_for_blocks raises ValueError when min_block_size is invalid.
    """
    with pytest.raises(ValueError):
        generator._get_start_indices_for_blocks(num_elements=100, num_blocks=5, min_block_size=0)
    with pytest.raises(ValueError):
        generator._get_start_indices_for_blocks(num_elements=100, num_blocks=5, min_block_size=-1)

def test_get_start_indices_num_elements_less_than_min_block_size(generator):
    """
    Test that _get_start_indices_for_blocks raises ValueError when num_elements is less than min_block_size.
    """
    with pytest.raises(ValueError):
        generator._get_start_indices_for_blocks(num_elements=4, num_blocks=1, min_block_size=5)

def test_get_start_indices_valid_inputs(generator):
    """
    Test that _get_start_indices_for_blocks returns correct number of unique start indices for valid inputs.
    """
    num_elements = 100
    num_blocks = 5
    min_block_size = 5
    start_indices = generator._get_start_indices_for_blocks(num_elements, num_blocks, min_block_size)
    assert len(start_indices) == num_blocks
    assert len(set(start_indices)) == num_blocks
    assert all(0 <= idx <= num_elements - min_block_size for idx in start_indices)

def test_generate_block_nans_invalid_percentage(generator):
    """
    Test that generate_block_nans raises ValueError when percentage is invalid.
    """
    with pytest.raises(ValueError):
        generator.generate_block_nans(percentage=-0.1, min_block_size=5, max_block_size=10)
    with pytest.raises(ValueError):
        generator.generate_block_nans(percentage=1.1, min_block_size=5, max_block_size=10)

def test_generate_block_nans_invalid_block_sizes(generator):
    """
    Test that generate_block_nans raises ValueError when block sizes are invalid.
    """
    with pytest.raises(ValueError):
        generator.generate_block_nans(percentage=0.1, min_block_size=0, max_block_size=10)
    with pytest.raises(ValueError):
        generator.generate_block_nans(percentage=0.1, min_block_size=-5, max_block_size=10)
    with pytest.raises(ValueError):
        generator.generate_block_nans(percentage=0.1, min_block_size=10, max_block_size=5)