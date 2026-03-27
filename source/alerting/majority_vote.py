from loguru import logger
from typing import List

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

    def vote(self, detections: List[int], use_threshold: bool = False, mode:str = "hard") -> int:
        """
        Perform majority voting.

        Parameters
        ----------
        detections : List[int]
            List of detections (0 or 1).
        use_threshold : bool
            If True, requires fraction of 1s >= self.threshold.
        mode : str
            "hard" for binary voting, "soft" for probabilistic voting (values in [0, 1]).

        Returns
        -------
        int
            1 if change point is detected, else 0.
        """
        if len(detections) == 0:
            raise ValueError("Detections list cannot be empty.")
        if mode not in ["hard", "soft"]:
            raise ValueError("Mode must be 'hard' or 'soft'.")
        if mode == "soft":
            if not all(0 <= d <= 1 for d in detections):
                raise ValueError("All detections must be in [0, 1] for soft voting.")
        else:
            if not all(d in (0, 1) for d in detections):
                raise ValueError("Hard voting expects binary detections (0 or 1).")

        n = len(detections)
        ratio = sum(detections)/n

        logger.info(f"mode={mode} total={n} ratio={ratio} threshold={self.threshold}")

        if mode == "soft" or (mode == "hard" and use_threshold):
            detected = ratio >= self.threshold
        else:  # hard
            detected = ratio > 0.5

        if detected:
            logger.warning(f"Change point detected (mode={mode}, ratio={ratio}).")
            return 1

        logger.info(f"No change point detected (mode={mode}, ratio={ratio}).")
        return 0
    
if __name__ == "__main__":

    mv = MajorityVote(threshold=0.6)
    detections = [1, 0, 1, 1, 0]
    result = mv.vote(detections, use_threshold=True, mode="hard")
    print("Majority vote result:", result)

    mv_soft = MajorityVote(threshold=0.7)
    soft_detections = [0.8, 0.6, 0.9, 0.4, 0.7]
    result_soft = mv_soft.vote(soft_detections, use_threshold=True, mode="soft")
    print("Soft majority vote result:", result_soft)
            