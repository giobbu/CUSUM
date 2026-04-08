import pytest
from source.alerting.majority_vote import MajorityVote

def test_threshold_below_0_5_raises_value_error():
    """Test that initializing MajorityVote with threshold below 0.5 raises ValueError."""
    with pytest.raises(ValueError):
        MajorityVote(threshold=0.4)

def test_threshold_above_1_raises_value_error():
    """Test that initializing MajorityVote with threshold above 1.0 raises ValueError."""
    with pytest.raises(ValueError):
        MajorityVote(threshold=1.5)

def test_valid_threshold_initialization():
    """Test that MajorityVote initializes correctly with valid threshold."""
    alert = MajorityVote(threshold=0.7)
    assert alert.threshold == 0.7