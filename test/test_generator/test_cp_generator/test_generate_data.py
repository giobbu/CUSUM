import numpy as np

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