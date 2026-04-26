import pytest

def test_vote_with_empty_detections(alert):
    """Test vote method with empty detections list."""
    with pytest.raises(ValueError):
        alert.vote([])

def test_vote_with_invalid_mode(alert):
    """Test vote method with invalid mode."""
    with pytest.raises(ValueError):
        alert.vote([0, 1], mode="invalid_mode")

def test_vote_with_invalid_detections_hard(alert):
    """Test vote method with invalid detections for hard mode."""
    with pytest.raises(ValueError):
        alert.vote([0, 1, 2], mode="hard")  # 2 is invalid

def test_vote_with_invalid_detections_soft(alert):
    """Test vote method with invalid detections for soft mode."""
    with pytest.raises(ValueError):
        alert.vote([0.5, 1.2], mode="soft")  # 1.2 is invalid

def test_vote_hard_mode_no_threshold(alert):
    """Test vote method in hard mode without threshold."""
    assert alert.vote([0, 0, 1]) == 0  # Ratio = 1/3 < 0.5
    assert alert.vote([0, 1, 1]) == 1  # Ratio = 2/3 > 0.5

def test_vote_hard_mode_with_threshold(alert):
    """Test vote method in hard mode with threshold."""
    alert.threshold = 0.7
    assert alert.vote([0, 0, 1], use_threshold=True) == 0  # Ratio = 1/3 < 0.7
    assert alert.vote([0, 1, 1], use_threshold=True) == 0  # Ratio = 2/3 < 0.7
    assert alert.vote([1, 1, 1], use_threshold=True) == 1  # Ratio = 3/3 >= 0.7

def test_vote_soft_mode(alert):
    """Test vote method in soft mode."""
    assert alert.vote([0.2, 0.5, 0.8], mode="soft") == 1  # Ratio = (0.2+0.5+0.8)/3 > 0.5
    assert alert.vote([0.2, 0.3, 0.4], mode="soft") == 0  # Ratio = (0.2+0.3+0.4)/3 < 0.5

def test_vote_soft_mode_with_threshold(alert):
    """Test vote method in soft mode with threshold."""
    alert.threshold = 0.6
    assert alert.vote([0.5, 0.6, 0.8], use_threshold=True, mode="soft") == 1  # Ratio = (0.2+0.5+0.8)/3 > 0.6
    assert alert.vote([0.2, 0.3, 0.4], use_threshold=True, mode="soft") == 0  # Ratio = (0.2+0.3+0.4)/3 < 0.6

def test_detected_return_type(alert):
    """Test that vote method returns an integer."""
    result = alert.vote([0, 1, 1])
    assert isinstance(result, int)

def test_detected_value(alert):
    """Test that vote method returns 0 or 1."""
    result = alert.vote([0, 1, 1])
    assert result == 1
    result = alert.vote([0, 0, 1])
    assert result == 0

def test_ratio_attribute(alert):
    """Test that ratio attribute is set correctly after voting."""
    alert.vote([0, 1, 1])
    assert alert.ratio == 2/3
    alert.vote([0.2, 0.5, 0.8], mode="soft")
    assert alert.ratio == (0.2+0.5+0.8)/3

def test_threshold_edge_cases(alert):
    """Test vote method with edge case thresholds."""
    alert.threshold = 0.5
    assert alert.vote([0, 1], use_threshold=True) == 1  # Ratio = 0.5 >= 0.5
    alert.threshold = 1.0
    assert alert.vote([1, 1], use_threshold=True) == 1  # Ratio = 1.0 >= 1.0
    assert alert.vote([0, 1], use_threshold=True) == 0  # Ratio = 0.5 < 1.0