import pytest
from source.alerting.majority_vote import MajorityVote

@pytest.fixture
def alert():
    """Fixture to initialize MajorityVote instance."""
    return MajorityVote()