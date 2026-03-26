from collections import Counter
from loguru import logger

class MajorityVote:
    """"
    MajorityVote class to perform majority voting on a list of binary detections.

    Parameters
    ----------
    threshold : float, optional
        The minimum proportion of detections required to classify as an anomaly (default is 0.5).
    """
    def __init__(self, threshold: float = 0.5):
        """
        Initialize the MajorityVote class.
        
        Parameters
        ----------
        threshold : float, optional
            The minimum proportion of detections required to classify as an anomaly (default is 0.5).
        """
        self.threshold = threshold

    def vote(self, detections):
        """
        Perform majority voting on the list of detections.
        
        Parameters
        ----------
        detections : list of int
            List of binary detections (0 for normal, 1 for anomaly).

        Returns
        -------
        int
            1 if anomaly is detected, 0 otherwise.
        """
        count = Counter(detections)
        most_common_label, most_common_count = count.most_common(1)[0]
        logger.info(f"Detections: {detections}, Most common label: {most_common_label}, Count: {most_common_count}")
        if most_common_count / len(detections) >= self.threshold:
            if most_common_label == 1:
                logger.warning("Anomaly detected based on majority vote.")
                return 1
            else:
                logger.info("No anomaly detected based on majority vote.")
                return 0
        else:
            return 0
            