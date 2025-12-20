from source.detector.cusum import CUSUM_Detector
import pytest

@pytest.fixture
def detector():
    """Fixture to initialize CUSUM_Detector instance with default parameters."""
    return CUSUM_Detector(warmup_period=10, delta=10, threshold=20)
