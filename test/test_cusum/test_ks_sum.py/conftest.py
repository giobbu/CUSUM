from source.detector.cusum import KS_CUM_Detector
import pytest

@pytest.fixture
def detector():
    """Fixture to initialize KS_CUM_Detector instance with default parameters."""
    return KS_CUM_Detector(window_pre=30, window_post=30, alpha=0.05)
