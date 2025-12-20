from source.detector.cusum import ProbCUSUM_Detector
import pytest

@pytest.fixture
def detector():
    """Fixture to initialize CUSUM_Detector instance with default parameters."""
    return ProbCUSUM_Detector(warmup_period=10, threshold_probability=0.01)
