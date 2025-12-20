from source.detector.cusum import ChartCUSUM_Detector
import pytest


@pytest.fixture
def detector():
    """Fixture to initialize ChartCUSUM_Detector instance."""
    return ChartCUSUM_Detector(warmup_period=10, level=3, deviation_type='dev')
