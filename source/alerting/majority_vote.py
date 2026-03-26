from collections import Counter
from loguru import logger

from collections import Counter
from typing import List, Optional

class MajorityVote:
    """
    Majority voting for change point detection (0 = no change point, 1 = change point).

    Parameters
    ----------
    threshold : float
        Minimum proportion of positive votes required when `use_threshold=True`.
        Must be in [0.5, 1.0].
    """

    def __init__(self, threshold: float = 0.5):
        if not 0.5 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.5 and 1.0.")
        self.threshold = threshold

    def vote(self, detections: List[int], use_threshold: bool = False) -> int:
        """
        Perform majority voting.

        Parameters
        ----------
        detections : List[int]
            List of detections (0 or 1).
        use_threshold : bool
            If True, requires fraction of 1s >= self.threshold.

        Returns
        -------
        int
            1 if change point is detected, else 0.
        """
        if len(detections) == 0:
            logger.warning("No detections provided. Returning 0.")
            return 0

        n = len(detections)
        positives = sum(detections)
        ratio = positives/n

        logger.info(
            f"Total={len(detections)}, count-positives={positives}, ratio={ratio:.3f}"
        )

        if use_threshold:
            if ratio >= self.threshold:
                logger.warning("Change point detected - threshold rule).")
                return 1
            logger.info("No change point detected - threshold rule).")
            return 0

        if positives > n/2:
            logger.warning("Change point detected - majority vote).")
            return 1

        logger.info("No change point detected - majority vote).")
        return 0
            