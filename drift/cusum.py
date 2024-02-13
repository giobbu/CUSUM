import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

class CUSUM_Detector:
    """
    CUSUM Change Point Detector Class

    Example:
    ```
    # Example usage of CUSUM_Detector
    detector = CUSUM_Detector(warmup_period=20, delta=15, threshold=30)
    data = [12.3, 14.5, 15.6, 16.8, 17.9, 20.2, 25.7, 30.2, 32.5, 32.9, 33.0, 32.2, 31.8, 30.5, 30.1]
    pos_changes, neg_changes, change_points = detector.detect_change_points(data)
    detector.plot_change_points(data, change_points, pos_changes, neg_changes)
    ```
    """

    def __init__(self, warmup_period=10, delta=10, threshold=20):
        """
        Initializes the Change Point Detector with the specified parameters.

        Parameters:
        - warmup_period (int): The number of initial observations before starting to detect change points. Default is 10.
        - delta (float): Sensitivity parameter for detecting changes. Default is 10.
        - threshold (float): Threshold for detecting a change point. Default is 20.
        """

        if not isinstance(warmup_period, int) or warmup_period < 10:
            raise ValueError("warmup_period must be equal or greater than 10.")

        self.warmup_period = warmup_period
        self.delta = delta
        self.threshold = threshold
        self._reset()

    def predict_next(self, observation):
        """
        Predicts the next data point and detects change points.

        Parameters:
        - observation (float): New data point.

        Returns:
        - pos_change (numpy array): Cumulative sum for positive changes.
        - neg_change (numpy array): Cumulative sum for negative changes.
        - is_changepoint (bool): Indicates if a change point is detected.
        """
        self._update_data(observation)
        if self.current_t == self.warmup_period:
            self._init_params()

        if self.current_t > self.warmup_period:
            self._compute_cumusum()
            is_changepoint = self._detect_changepoint()
            if is_changepoint:
                self._reset()
            return self.pos_change, self.neg_change, is_changepoint
        else:
            return self.pos_change, self.neg_change, False

    def _reset(self):
        """Resets the internal state of the detector."""
        self.current_t = 0
        self.current_obs = []
        self.current_mean = None
        self.current_std = None
        self.z = None
        self.pos_change = np.array([0.0])
        self.neg_change = np.array([0.0])
        self.drift = []

    def _update_data(self, observation):
        """Updates the observed data with new data points."""
        self.current_t += 1
        self.current_obs.append(observation)

    def _init_params(self):
        """Initializes the parameters required for CUSUM computation."""

        self.current_mean = np.nanmean(np.array(self.current_obs))
        self.current_std = np.nanstd(np.array(self.current_obs))
        self.cumulative_sum_pos = [0.0]
        self.cumulative_sum_neg = [0.0]
        if math.isnan(self.current_mean) or math.isnan(self.current_std):
            raise ValueError("Mean or standard deviation cannot be NaN")

    def _compute_cumusum(self):
        """Computes the cumulative sums for positive and negative changes."""
        self.z = (self.current_obs[-1] - self.current_mean) / self.current_std  
        self.pos_change = max(0, self.cumulative_sum_pos[-1] + self.z - self.delta) 
        self.neg_change = max(0, self.cumulative_sum_neg[-1] - self.z - self.delta) 
        self.cumulative_sum_pos.append(self.pos_change)
        self.cumulative_sum_neg.append(self.neg_change)

    def _detect_changepoint(self):
        """
        Detects change points based on the computed cumulative sums.

        Returns:
        - is_changepoint (bool): Indicates if a change point is detected.
        """
        if self.cumulative_sum_pos[-1] > self.threshold or self.cumulative_sum_neg[-1] > self.threshold:
            self.drift.append(self.current_t)
            return True
        else:
            return False

    def detect_change_points(self, data):
        """
        Detects change points in the given data using the CUSUM detector.

        Parameters:
        - data (numpy array): Data points to be analyzed.

        Returns:
        - pos_changes (numpy array): Positive cumulative sum values.
        - neg_changes (numpy array): Negative cumulative sum values.
        - change_points (numpy array): Detected change points indices.
        """

        if not isinstance(data, np.ndarray):
            raise ValueError("data must be a numpy array.")
        if len(data) < self.warmup_period:
            raise ValueError("Data length must be greater than or equal to warmup_period.")

        outs = [self.predict_next(point) if not math.isnan(point) else (0, 0, False) for point in data]
        pos_changes = np.vstack([row[0] for row in outs])
        neg_changes = np.vstack([row[1] for row in outs])
        is_drift = [row[2] for row in outs]
        change_points = np.array([i for i, drift in enumerate(is_drift) if drift])

        return pos_changes, neg_changes, change_points

    def plot_change_points(self, data, change_points, pos_changes, neg_changes):
        """
        Plots data with detected change points and cumulative sums.

        Parameters:
        - data (numpy array): Original data points.
        - change_points (list): List of detected change points.
        - pos_changes (list): List of positive cumulative sum values.
        - neg_changes (list): List of negative cumulative sum values.
        """
        plt.figure(figsize=(20, 8))

        plt.subplot(2, 1, 1)
        plt.plot(data, color='blue', label='Data', linestyle="--")
        plt.axvline(change_points[0], color="red", linestyle="dashed", label='Change Points', lw=2)
        [plt.axvline(cp, color="red", linestyle="dashed", lw=2) for cp in change_points[1:]]
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Sequential CUSUM Change Point Detection')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(pos_changes, color='green', label='Positive Cumulative Sum')
        plt.plot(neg_changes, color='orange', label='Negative Cumulative Sum')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Sum')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()



class ProbCUSUM_Detector:
    """
    A class to detect change points in sequential data using the Probabilistic Cumulative Sum (CUSUM) algorithm.

    Example:
    ```
    # Example usage of ProbCUMSUM_Detector
    detector = ProbCUMSUM_Detector(warmup_period=10, threshold_probability=0.001)
    data = [10.2, 11.5, 12.6, 12.8, 12.9, 13.2, 12.7, 12.5, 12.3, 12.9, 25.0, 12.2, 11.8, 10.5, 10.1]
    probabilities, change_points = detector.detect_change_points(data)
    detector.plot_change_points(data, change_points, probabilities)
    ```
    """

    def __init__(self, warmup_period=10, threshold_probability=0.001):
        """
        Initializes the Probabilistic CUSUM Detector with the specified parameters.

        Parameters:
        - warmup_period (int): The number of initial observations before starting to detect change points. Default is 10.
        - threshold_probability (float): The threshold probability below which a change point is detected. Default is 0.001.
        """

        if not isinstance(warmup_period, int) or warmup_period < 10:
            raise ValueError("warmup_period must be equal or greater than 10.")
        if not isinstance(threshold_probability, float) or threshold_probability <= 0 or threshold_probability >= 1:
            raise ValueError("threshold_probability must be a float between 0 and 1.")

        self.warmup_period = warmup_period
        self.threshold_probability = threshold_probability
        self.running_sum = 0  # Initialize running sum of standardized observations
        self._reset()
        
    def predict_next(self, observation):
        """
        Predicts the probability of a change point in the next observation.

        Parameters:
        - observation (float): The next observation in the sequence.

        Returns:
        - probability (float): The probability of a change point in the next observation.
        - is_changepoint (bool): True if a change point is detected, False otherwise.
        """
        self._update_data(observation)
        if self.current_t == self.warmup_period:
            self._init_params()
        if self.current_t >= self.warmup_period:
            probability, is_changepoint = self._detect_changepoint()
            if is_changepoint:
                self._reset()
            return (1 - probability), is_changepoint
        else:
            return 0, False
    
    def _reset(self) -> None:
        """
        Resets the internal state of the detector.
        """
        self.current_t = 0
        self.observations = []
        self.mean_observation = None
        self.std_dev_observation = None
        self.running_sum = 0  # Reset running sum
    
    def _update_data(self, observation) -> None:
        """
        Updates the internal state with a new observation.

        Parameters:
        - observation (float): The new observation to be added.
        """
        self.current_t += 1
        self.observations.append(observation)
        
    def _init_params(self) -> None:
        """
        Initializes the mean and standard deviation of observations.
        """

        if len(self.observations) < 2:
            raise ValueError("At least two observations are needed to initialize parameters.")

        self.mean_observation = np.nanmean(np.array(self.observations))
        self.std_dev_observation = np.nanstd(np.array(self.observations))
    
    def _detect_changepoint(self):
        """
        Detects a change point using the CUSUM algorithm.

        Returns:
        - probability (float): The probability of a change point.
        - is_changepoint (bool): True if a change point is detected, False otherwise.
        """
        self.running_sum += (self.observations[-1] - self.mean_observation)  # Update running sum
        standardized_sum = self.running_sum / (self.std_dev_observation * self.current_t**0.5)
        probability = float(self._calculate_probability(standardized_sum))
        return probability, probability < self.threshold_probability
    
    def _calculate_probability(self, standardized_sum) -> bool:
        """
        Calculates the probability of a change point.

        Parameters:
        - standardized_sum (float): The standardized sum of observations.

        Returns:
        - probability (float): The probability of a change point.
        """
        p_obs = norm.cdf(np.abs(standardized_sum))
        probability = 2 * (1 - p_obs)
        return probability
    
    def detect_change_points(self, data):
        """
        Detects change points in the given data using the CUSUM detector.

        Parameters:
        - data: numpy array
            Data points to be analyzed.

        Returns:
        - probabilities: numpy array
            Probability values for each data point.
        - change_points: numpy array
            Detected change points indices.
        """

        if not isinstance(data, np.ndarray):
            raise ValueError("data must be a numpy array.")
        if len(data) < self.warmup_period:
            raise ValueError("Data length must be greater than or equal to warmup_period.")

        results = [self.predict_next(point) if not math.isnan(point) else (0, False) for point in data]
        probabilities = np.array([result[0] for result in results])
        is_drift = np.array([result[1] for result in results])
        change_points = np.where(is_drift)[0]

        return probabilities, change_points

    def plot_change_points(self, data, change_points, probabilities):
        """
        Plots data with detected change points and probabilities.

        Parameters:
        - data: numpy array
            Original data points.
        - change_points: list
            List of detected change points.
        - probabilities: list
            List of probabilities associated with each data point.
        """
        plt.figure(figsize=(20, 8))

        plt.subplot(2, 1, 1)
        plt.plot(data, color='blue', label='Data', linestyle="--")
        X, Y = np.meshgrid(np.arange(len(data)), np.linspace(0, max(data)))
        Z = probabilities[X]
        plt.contourf(X, Y, Z, alpha=0.1, cmap="Reds")
        for cp in change_points:
            plt.axvline(cp, color="red", linestyle="dashed", lw=2, label='Change Points' if cp == change_points[0] else None)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Sequential Probabilistic CUSUM Change Point Detection')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(probabilities, color='red', label='Probability', linestyle="dashed")
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()