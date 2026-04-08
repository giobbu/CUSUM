import pytest
import numpy as np

# from loguru import logger
# from typing import List

# class MajorityVote:
#     """
#     Majority voting for change point detection (0 = no change point, 1 = change point).

#     Parameters
#     ----------
#     threshold : float
#         Minimum proportion of positive votes required when `use_threshold=True`.
#         Must be in [0.5, 1.0].
#     """

#     def __init__(self, threshold: float = 0.5):
#         if not 0.5 <= threshold <= 1.0:
#             raise ValueError("Threshold must be between 0.5 and 1.0.")
#         self.threshold = threshold

#     def vote(self, detections: List[int], use_threshold: bool = False, mode:str = "hard") -> int:
#         """
#         Perform majority voting.

#         Parameters
#         ----------
#         detections : List[int]
#             List of detections (0 or 1).
#         use_threshold : bool
#             If True, requires fraction of 1s >= self.threshold.
#         mode : str
#             "hard" for binary voting, "soft" for probabilistic voting (values in [0, 1]).

#         Returns
#         -------
#         int
#             1 if change point is detected, else 0.
#         """
#         if len(detections) == 0:
#             raise ValueError("Detections list cannot be empty.")
#         if mode not in ["hard", "soft"]:
#             raise ValueError("Mode must be 'hard' or 'soft'.")
#         if mode == "soft":
#             if not all(0 <= d <= 1 for d in detections):
#                 raise ValueError("All detections must be in [0, 1] for soft voting.")
#         else:
#             if not all(d in (0, 1) for d in detections):
#                 raise ValueError("Hard voting expects binary detections (0 or 1).")

#         n = len(detections)
#         ratio = sum(detections)/n

#         logger.info(f"mode={mode} total={n} ratio={ratio} threshold={self.threshold}")

#         if mode == "soft" or (mode == "hard" and use_threshold):
#             detected = ratio >= self.threshold
#         else:  # hard
#             detected = ratio > 0.5

#         if detected:
#             logger.warning(f"Change point detected (mode={mode}, ratio={ratio}).")
#             return 1

#         logger.info(f"No change point detected (mode={mode}, ratio={ratio}).")
#         return 0

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