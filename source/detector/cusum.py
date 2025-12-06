import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

class CUSUM_Detector:
    """
    CUSUM Change Point Detector Class.
    
    Example:
    ```
    detector = CUSUM_Detector(warmup_period=10, delta=10, threshold=20)
    data = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])
    pos_changes, neg_changes, change_points = detector.offline_detection(data)
    detector.plot_change_points(data, change_points, pos_changes, neg_changes)
    ```
    """
    def __init__(self, warmup_period:int=10, delta:int=10, threshold:int=20):
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

    def detection(self, observation:float):
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
        if self.current_t < self.warmup_period:
            self.S_pos = np.array([0])
            self.S_neg = np.array([0])
        if self.current_t == self.warmup_period:
            self._init_params()
        if self.current_t > self.warmup_period:
            self._compute_cumusum()
            is_changepoint = self._detect_changepoint()
            if is_changepoint:
                self._reset()
            return self.S_pos, self.S_neg, is_changepoint
        else:
            return self.S_pos, self.S_neg, False

    def _reset(self):
        """Resets the internal state of the detector."""
        self.current_t = 0
        self.current_mean = 0
        self.current_std = 0
        self.current_obs = []

    def _update_data(self, observation: float):
        """Updates the observed data with new data points."""
        self.current_t += 1
        self.current_obs.append(observation)

    def _init_params(self):
        """Initializes the parameters required for CUSUM computation."""
        self.current_mean = np.nanmean(np.array(self.current_obs))
        self.current_std = np.nanstd(np.array(self.current_obs))
        self.z = 0
        self.S_pos = np.array([0])
        self.S_neg = np.array([0])
        # if math.isnan(self.current_mean) or math.isnan(self.current_std):
        #     raise ValueError("Mean or standard deviation cannot be NaN")

    def _compute_cumusum(self):
        """Computes the cumulative sums for positive and negative changes."""
        self.z = (self.current_obs[-1] - self.current_mean) / self.current_std  
        self.S_pos = max(np.array([0]), self.S_pos + self.z - self.delta) 
        self.S_neg = max(np.array([0]), self.S_neg - self.z - self.delta) 

    def _detect_changepoint(self):
        """
        Detects change points based on the computed cumulative sums.

        Returns:
        - is_changepoint (bool): Indicates if a change point is detected.
        """
        if self.S_pos > self.threshold or self.S_neg > self.threshold:
            return True
        else:
            return False

    def offline_detection(self, data: np.ndarray):
        """
        Detects change points in the given data in an offline manner.

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
        
        results = [self.detection(point) if not math.isnan(point) else (np.array([0]), np.array([0]), False) for point in data]
        pos_changes = np.vstack([row[0] for row in results])
        neg_changes = np.vstack([row[1] for row in results])
        is_drift = [row[2] for row in results]
        change_points = np.array([i for i, drift in enumerate(is_drift) if drift])

        return pos_changes, neg_changes, change_points

    def plot_change_points(self, data: np.ndarray, change_points: list, pos_changes: list, neg_changes: list):
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
        if len(change_points) != 0:
            plt.axvline(change_points[0], color="red", linestyle="dashed", label='Change Points', lw=2)
            [plt.axvline(cp, color="red", linestyle="dashed", lw=2) for cp in change_points[1:]]
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Sequential CUSUM Change Point Detection')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.axhline(self.threshold , color="red", linestyle="dashed", lw=2)
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
    detector = ProbCUSUM_Detector(warmup_period=10, threshold_probability=0.01)
    data = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])
    probabilities, change_points = detector.offline_detection(data)
    detector.plot_change_points(data, change_points, probabilities)
    ```
    """

    def __init__(self, warmup_period:int=10, threshold_probability:float=0.001):
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
        
    def detection(self, observation: float):
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
        if self.current_t > self.warmup_period:
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
    
    def _update_data(self, observation:float) -> None:
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
    
    def _calculate_probability(self, standardized_sum:float) -> bool:
        """
        Calculates the probability (p-value) of a change point.

        Parameters:
        - standardized_sum (float): The standardized sum of observations.

        Returns:
        - probability (float): The probability of a change point.
        * low probability indicates a change point (reject the null hypothesis).
        * high probability indicates no change point (not able to reject the null hypothesis).
        """
        p_obs = norm.cdf(np.abs(standardized_sum))
        probability = 2 * (1 - p_obs)
        return probability
    
    def offline_detection(self, data: np.ndarray):
        """
        Detects change points in the given data in an offline manner.

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
        
        results = [self.detection(point) if not math.isnan(point) else (0, False) for point in data]
        probabilities = np.array([result[0] for result in results])
        is_drift = np.array([result[1] for result in results])
        change_points = np.where(is_drift)[0]

        return probabilities, change_points

    def plot_change_points(self, data: np.ndarray, change_points: list, probabilities: list):
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
        # Plot the data
        plt.subplot(2, 1, 1)
        plt.plot(data, color='blue', label='Data', linestyle="--")
        X, Y = np.meshgrid(np.arange(len(data)), np.linspace(0, max(data)))
        Z = probabilities[X]
        plt.contourf(X, Y, Z, alpha=0.1, cmap="Greys")
        if len(change_points) != 0:
            for cp in change_points:
                plt.axvline(cp, color="red", linestyle="dashed", lw=2, label='Change Points' if cp == change_points[0] else None)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Sequential Probabilistic CUSUM Change Point Detection')
        plt.legend()
        plt.grid(True)
        # Plot the probabilities
        plt.subplot(2, 1, 2)
        plt.axhline((1-self.threshold_probability), color="red", alpha=0.5, linestyle="dashed", lw=2)
        plt.plot(probabilities, color='gray', alpha=0.5, label='Alert Probability')
        if len(change_points) != 0:
            for cp in change_points:
                plt.axvline(cp, color="red", alpha=0.5, linestyle="dashed", lw=2, label='Change Points' if cp == change_points[0] else None)
        plt.xlabel('Time')
        plt.ylabel('Alert Probability')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class ChartCUSUM_Detector:

    """
    Change Point Detector using CUSUM Control Chart.

    Example:
    ```
    detector = ChartCUSUM_Detector(warmup_period=20, level=3, deviation_type='sqr-dev')
    data = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])
    upper_limits, lower_limits, cusums, change_points = detector.offline_detection(data)
    detector.plot_change_points(data, change_points, cusums, upper_limits, lower_limits)
    ```
    """
    def __init__(self, warmup_period:int=10, level:int=3, deviation_type:str='sqr-dev', target_mean:float=None):
        """
        Initializes the Change Point Detector with the specified parameters.
        
        Parameters:
        - warmup_period (int): The warmup period for the detector. Must be equal or greater than 10.
        - level (int): The level parameter for the CUSUM algorithm.
        - type (str): The type of deviation used in CUSUM algorithm. 'sqr-dev' for square deviation, else deviation.
        """

        if not isinstance(warmup_period, int) or warmup_period < 10:
            raise ValueError("warmup_period must be equal or greater than 10.")
        
        if level<1 or level>3:
            raise ValueError("level must be between 1 and 3.")
        
        if deviation_type not in ['sqr-dev', 'dev']:
            raise ValueError("deviation_type must be 'sqr-dev' or 'dev'.")
        
        self.warmup_period = warmup_period
        self.level = level
        self.deviation_type = deviation_type
        self.target_mean = target_mean
        self._reset()

    def detection(self, observation: float):
        """
        Predicts the next data point and detects change points.
        
        Parameters:
        - observation (float): The next data point to predict.

        Returns:
        - upper (float): The upper limit of the CUSUM.
        - lower (float): The lower limit of the CUSUM.
        - cusum (float): The current value of the CUSUM.
        - is_changepoint (bool): Indicates if a change point is detected.
        """
        self._update_data(observation)
        if self.current_t == self.warmup_period:
            self._init_chart_stats()
        if self.current_t > self.warmup_period:
            self._update_chart_stats()
            is_changepoint = self._detect_changepoint()
            if is_changepoint:
                self._reset()
            return self.upper, self.lower, self.cusum, is_changepoint
        else:
            return self.upper, self.lower, self.cusum, False

    def _reset(self):
        """
        Resets the internal state of the detector.
        """
        self.current_t = 0
        self.current_obs = []
        self.cusum = 0
        self.upper = 0
        self.lower = 0

    def _update_data(self, observation: float):
        """
        Updates the observed data with new data points.
        
        Parameters:
        - observation (float): The new data point to update.
        """
        self.current_t += 1
        self.current_obs.append(observation)

    def _init_chart_stats(self):
        """
        Initializes the parameters required for CUSUM computation.
        """
        if not self.target_mean:
            self.window_mean = np.nanmean(np.array(self.current_obs))
        else:
            self.window_mean = self.target_mean

        if self.deviation_type == 'sqr-dev':
            self.warmup_cusum = np.nancumsum((np.array(self.current_obs) - self.window_mean) ** 2)
        else:
            self.warmup_cusum = np.nancumsum(np.array(self.current_obs) - self.window_mean)
        self.cusum = 0
        self.cusum_mean = 0
        self.cusum_std = 0
        self.upper = 0
        self.lower = 0

    def _update_chart_stats(self):
        """
        Updates the chart statistics after receiving a new data point.
        """
        if not self.target_mean:
            self.window_mean = np.nanmean(np.array(self.current_obs[1:]))
        else:
            self.window_mean = self.target_mean
            
        if self.deviation_type == 'sqr-dev':
            self.cusum = self.cusum + ((self.current_obs[-1] - self.window_mean) ** 2)
        else:
            self.cusum = self.cusum + (self.current_obs[-1] - self.window_mean)
        self.warmup_cusum = np.append(self.warmup_cusum[1:], self.cusum)
        self.cusum_mean = np.nanmean(np.array(self.warmup_cusum))
        self.cusum_std = np.nanstd(np.array(self.warmup_cusum))
        self.upper = self.cusum_mean + self.level * self.cusum_std
        self.lower = self.cusum_mean - self.level * self.cusum_std

    def _detect_changepoint(self):
        """ 
        Detects change points based on the computed cumulative sums.
        
        Returns:
        - bool: Indicates if a change point is detected.
        """
        if self.cusum > self.upper or self.cusum < self.lower:
            return True
        else:
            return False

    def offline_detection(self, data: np.ndarray):
        """
        Detects change points in the given data in an offline manner.
        
        Parameters:
        - data (np.ndarray): The data to detect change points in.

        Returns:
        - upper_limits (np.ndarray): Upper limits of the CUSUM for each observation.
        - lower_limits (np.ndarray): Lower limits of the CUSUM for each observation.
        - cusums (np.ndarray): Cumulative sums of deviations.
        - change_points (np.ndarray): Indices of detected change points.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("data must be a numpy array.")
        if len(data) < self.warmup_period:
            raise ValueError("Data length must be greater than or equal to warmup_period.")
        results = [self.detection(point) if not math.isnan(point) else (0, 0, 0, False) for point in data]
        upper_limits = np.vstack([row[0] for row in results])
        lower_limits = np.vstack([row[1] for row in results])
        cusums = np.vstack([row[2] for row in results])
        is_drift = [row[3] for row in results]
        change_points = np.array([i for i, drift in enumerate(is_drift) if drift])
        return upper_limits , lower_limits , cusums, change_points

    def plot_change_points(self, data:np.ndarray, change_points:list, cusums:list, upper_limits:list, lower_limits:list):
        """
        Plots data with detected change points and CUSUM statistics.

        Parameters:
        - data (np.ndarray): Original data points.
        - change_points (list): List of detected change points.
        - cusums (list): List of cumulative sums.
        - upper_limits (list): List of upper limits.
        - lower_limits (list): List of lower limits.
        """
        plt.figure(figsize=(20, 8))
        plt.subplot(2, 1, 1)
        plt.plot(data, color='blue', label='Data', linestyle="--")
        if len(change_points) != 0:
            plt.axvline(change_points[0], color="red", linestyle="dashed", label='Change Points', lw=2)
            [plt.axvline(cp, color="red", linestyle="dashed", lw=2) for cp in change_points[1:]]
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Control Chart CUSUM Change Point Detection')
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(cusums, color='green', label= f'Cumulative Sum of {self.deviation_type}')
        plt.plot(upper_limits, color='red', linestyle="dashed", label='Upper Limit')
        plt.plot(lower_limits, color='red', linestyle="dashed", label='Lower Limit')
        plt.xlabel('Time')
        plt.ylabel(f'Cumulative Sum of {self.deviation_type}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class KS_CUM_Detector:
    """ 
    A class to detect change points in sequential data using the Kolmogorov-Smirnov Test, loosley named Kolmogorov-Smirnov 
    Cumulative Sum (KS-CUM) algorithm.
    Example:
    ```
    detector = KS_CUM_Detector(window_pre=30, window_post=30, alpha=0.05)
    data = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])
    ks_statistics, p_values, change_points = detector.offline_detection(data)
    detector.plot_change_points(data, change_points, p_values)
    ```
    Args:
    - window_pre (int): The size of the pre-change window.
    - window_post (int): The size of the post-change window.
    - alpha (float): The significance level for the KS test.
    Returns:
    - ks_statistics (np.ndarray): The KS statistics for each observation.
    - p_values (np.ndarray): The p-values for each observation.
    - change_points (np.ndarray): The indices of detected change points.
    """

    def __init__(self, window_pre:int=30, window_post:int=30, alpha:float=0.05):
        assert window_pre >= 30, "window_pre must be greater than 30."
        assert window_post >= 30, "window_post must be greater than 30."
        assert window_pre >= window_post, "window_pre must be greater than or equal to window_post."
        assert 0 < alpha < 0.1, "alpha must be between 0 and 0.1."
        self.window_pre = window_pre
        self.window_post = window_post
        self.warmup_period = window_pre + window_post
        self.alpha = alpha
        self._reset()

    def detection(self, observation: float):
        " Predicts the next data point and detects change points."
        self._update_data(observation)
        is_changepoint = False
        if self.current_t < self.warmup_period:
            self.ks_statistic = np.array([0])
            self.p_value = np.array([1])
        if self.current_t >= self.warmup_period:
            self._init_params()
            self._compute_KS_statistic()
            is_changepoint = self._detect_changepoint()
            if is_changepoint:
                self._reset()
            return self.ks_statistic, self.p_value, is_changepoint
        else:
            return self.ks_statistic, self.p_value, is_changepoint
        
    def _init_params(self):
        " Initializes the parameters required for KS-CUM computation."
        self.window_pre_data = self.current_obs[-self.warmup_period: -self.warmup_period + self.window_pre]
        self.window_post_data = self.current_obs[-self.window_post:]

    def _reset(self):
        " Resets the internal state of the detector."
        self.current_t = 0
        self.current_obs = []

    def _update_data(self, observation: float):
        " Updates the observed data with new data points."
        # # skip if observation is NaN
        # if math.isnan(observation):
        #     return
        self.current_t += 1
        self.current_obs.append(observation)

    def _compute_KS_statistic(self):
        " Computes the Kolmogorov-Smirnov statistic for the current window."
        from scipy import stats
        self.ks_statistic, self.p_value = stats.ks_2samp(self.window_pre_data, self.window_post_data)

    def _detect_changepoint(self):
        " Detects change points based on the computed KS statistic."
        if self.p_value < self.alpha:
            return True
        else:
            return False

    def offline_detection(self, data: np.ndarray):
        " Detects change points in the given data in an offline manner."
        if not isinstance(data, np.ndarray):
            raise ValueError("data must be a numpy array.")
        results = [self.detection(point) if not math.isnan(point) else (np.array([0]), np.array([0]), False) for point in data]
        ks_statistics = np.vstack([row[0] for row in results])
        p_values = np.vstack([row[1] for row in results])
        is_changepoint = [row[2] for row in results]
        change_points = np.array([i for i, drift in enumerate(is_changepoint) if drift])
        return ks_statistics , p_values, change_points

    def plot_change_points(self, data: np.ndarray, change_points: list, p_values: list):
        """ 
        Plots data with detected change points and KS statistics.

        Parameters:
        - data (numpy array): Original data points.
        - change_points (list): List of detected change points.
        - p_values (list): List of p-values associated with each data point.
        """
        plt.figure(figsize=(20, 8))
        plt.subplot(2, 1, 1)
        plt.plot(data, color='blue', label='Data', linestyle="--")
        X, Y = np.meshgrid(np.arange(len(data)), np.linspace(0, max(data)))  # create a grid
        Z = 1 - p_values[X]  # get the p-values for the grid
        plt.contourf(X, Y, Z[:,:,0], alpha=0.1, cmap="Greys")
        if len(change_points) != 0:
            for cp in change_points:
                plt.axvline(cp, color="red", linestyle="dashed", label=f'Change Points-{cp}', lw=2)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('KS-Test Change Point Detection')
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(1-p_values, color='gray', label='(1-P-Values)')
        # plot horizontal line at alpha
        plt.axhline(1-self.alpha, color='red', linestyle='dashed', label='(1-Alpha)')
        plt.xlabel('Time')
        plt.ylabel('Probability of Change')
        plt.title('Probability of Changepoint')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()