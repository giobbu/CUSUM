import pytest
from source.detector.cusum import KS_CUM_Detector

def test_init_with_invalid_window_pre():
    """Test initialization with invalid window_pre."""
    with pytest.raises(ValueError):
        KS_CUM_Detector(window_pre=30.1, window_post=30, alpha=0.05)

def test_init_with_invalid_window_post():
    """Test initialization with invalid window_post."""
    with pytest.raises(ValueError):
        KS_CUM_Detector(window_pre=35, window_post=30.5, alpha=0.05)

def test_init_with_lower_pre_window_size():
    """Test initialization with non-integer window sizes."""
    with pytest.raises(ValueError):
        KS_CUM_Detector(window_pre=25, window_post=30, alpha=0.05)

def test_init_with_lower_post_window_size():
    """Test initialization with non-integer window sizes."""
    with pytest.raises(ValueError):
        KS_CUM_Detector(window_pre=30, window_post=25, alpha=0.05)

def test_init_with_pre_lower_than_post():
    """Test initialization with window_pre less than window_post."""
    with pytest.raises(ValueError):
        KS_CUM_Detector(window_pre=20, window_post=30, alpha=0.05)

def test_init_with_invalid_alpha():
    """Test initialization with invalid alpha."""
    with pytest.raises(ValueError):
        KS_CUM_Detector(window_pre=30, window_post=30, alpha=1.5)
