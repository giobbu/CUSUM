import numpy as np
from source.generator.change_point_generator import ChangePointGenerator
from source.generator.ds_generator import MultiDataStreams
import pytest

def test_init_with_invalid_num_streams():
    """Test initialization with invalid num_streams."""
    with pytest.raises(AssertionError):
        MultiDataStreams(num_streams=2, dict_streams=[{"seed": 42}])

def test_init_with_valid_num_streams():
    """Test initialization with valid num_streams."""
    mds = MultiDataStreams(num_streams=2, dict_streams=[{"seed": 42}, {"seed": 43}])
    assert len(mds.list_generators) == 2
    assert isinstance(mds.list_generators[0], ChangePointGenerator)
    assert isinstance(mds.list_generators[1], ChangePointGenerator)

def test_init_with_default_parameters():
    """Test initialization with default parameters."""
    mds = MultiDataStreams()
    assert len(mds.list_generators) == 2
    assert isinstance(mds.list_generators[0], ChangePointGenerator)
    assert isinstance(mds.list_generators[1], ChangePointGenerator)

def test_attributes_assigned_correctly():
    """Test that attributes are assigned correctly."""
    mds = MultiDataStreams()
    assert mds.dict_streams == [None, None]
    assert mds.list_data_streams == []
    assert mds.dict_missing is None

