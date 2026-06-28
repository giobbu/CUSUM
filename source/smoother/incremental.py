import numpy as np

class RecursiveAverage:
    """
    Recursive Average Filter for incremental average computation.
    """
    def __init__(self) -> None:
        """
        Initialize the RecursiveAverage object.
        """
        self.recursive_mean = None
        self.num_iterations = 0

    def update(self, observation: np.ndarray) -> None:
        """
        Update the recursive mean with a new observation.
        
        Parameters
        ----------
        observation : np.ndarray
            New observation to update the mean.
        """
        if not isinstance(observation, np.ndarray):
            raise ValueError("Observation must be a numpy array.")
        if observation.ndim != 1:
            raise ValueError("Observation must be a 1D numpy array.")
        
        self.num_iterations += 1
        if self.recursive_mean is None:
            self.recursive_mean = observation
        else:
            alpha = (self.num_iterations-1)/self.num_iterations  # alpha depends on the number of points, it is not free
            self.recursive_mean = alpha*self.recursive_mean + (1-alpha)*observation

    def fit(self, observations: list[np.ndarray]) -> None:
        """
        Fit the recursive mean to a set of observations.
        
        Parameters
        ----------
        observations : list of np.ndarray
            List of observation vectors.

        Returns
        -------
        list_smooth : list of np.ndarray
            List of recursive means after each observation.
        """
        if not isinstance(observations, list):
            raise ValueError("Observations must be a list of numpy arrays.")
        
        list_smooth = []
        for observation in observations:
            self.update(observation)
            list_smooth.append(self.recursive_mean)
        return list_smooth

class LowPassFilter:
    """
    Low-Pass Filter using Exponential Moving Average.

    Parameters
    ----------
    alpha : float, optional
        Smoothing factor between 0 and 1.
    """
    def __init__(self, alpha: float = 0.999) -> None:
        """
        Initialize the LowPassFilter object.
        
        Parameters
        ----------
        alpha : float, optional
            Smoothing factor between 0 and 1.
        """
        if alpha <= 0 or alpha >= 1:
            raise ValueError("Alpha must be between 0 and 1.")
        
        self.lowpass_mean = None
        self.num_iterations = 0
        self.alpha = alpha

    def update(self, observation: np.ndarray) -> None:
        """
        Update the low-pass mean with a new observation.
        
        Parameters
        ----------
        observation : np.ndarray
            New observation to update the mean.
        """
        if not isinstance(observation, np.ndarray):
            raise ValueError("Observation must be a numpy array.")
        if observation.ndim != 1:
            raise ValueError("Observation must be a 1D numpy array.")
        
        self.num_iterations += 1
        if self.lowpass_mean is None:
            self.lowpass_mean = observation
        else:
            self.lowpass_mean = self.alpha*self.lowpass_mean + (1-self.alpha)*observation

    def fit(self, observations: list[np.ndarray]) -> None:
        """
        Fit the low-pass mean to a set of observations.

        Parameters
        ----------
        observations : list of np.ndarray
            List of observation vectors.
        
        Returns
        -------
        list_smooth : list of np.ndarray
            List of low-pass means after each observation.
        """
        if not isinstance(observations, list):
            raise ValueError("Observations must be a list of numpy arrays.")
        
        list_smooth = []
        for observation in observations:
            self.update(observation)
            list_smooth.append(self.lowpass_mean)
        return list_smooth
    


class RollingAverageFilter:
    """ 
    Rolling Average Filter algorithm for filtering out high frequency noise.

    Parameters
    ----------
    window : int, optional
        Size of the moving window. 
    """
    def __init__(self, window: int = 3) -> None:
        """ 
        Initialize the RollingAverageFilter object. 
        
        Parameters
        ----------
        window : int, optional
            Size of the moving window. 
        """
        if window <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.moving_mean = None
        self.num_iterations = 0
        self.window = window
        self.rolling_list = []

    def update(self, observation: np.ndarray) -> None:
        """ 
        Update the model with the new observation.

        Parameters
        ----------
        observation : np.ndarray
            New observation to update the mean.
        """
        if not isinstance(observation, np.ndarray):
            raise ValueError("Observation must be a numpy array.")
        if observation.ndim != 1:
            raise ValueError("Observation must be a 1D numpy array.")
        
        self.num_iterations += 1
        self.rolling_list.append(observation)  # always append first
        if self.moving_mean is None:  # first observation
            self.moving_mean = observation.copy()
        elif len(self.rolling_list) <= self.window:  # warm-up: mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n
            self.moving_mean += (observation - self.moving_mean) / len(self.rolling_list)
        else:
            del self.rolling_list[0]  # remove the oldest observation to maintain the window size
            self.moving_mean = np.mean(self.rolling_list[-self.window:], axis=0)  # full window: mean_n = mean_{n-1} + (x_n - x_{n-window}) / window

    
    def fit(self, observations: list[np.ndarray]):
        """ 
        Fit the model to a set of observations.

        Parameters
        ----------
        observations : list of np.ndarray
            List of observation vectors.

        Returns
        -------
        list_smooth : list of np.ndarray
            List of moving means after each observation.
        """
        list_smooth = []
        for observation in observations:
            self.update(observation)
            list_smooth.append(self.moving_mean)
        return list_smooth
    



class KalmanFilter:
    """
    Kalman Filter algorithm for filtering out noise in linear dynamical systems.

    Parameters
    ----------
    F : np.ndarray
        State transition matrix of shape (n, n).
    H : np.ndarray
        Observation matrix of shape (m, n).
    Q : np.ndarray
        Process noise covariance matrix of shape (n, n).
    R : np.ndarray
        Measurement noise covariance matrix of shape (m, m).
    """
    def __init__(self, F: np.ndarray, H: np.ndarray,
                 Q: np.ndarray, R: np.ndarray) -> None:
        """
        Initialize the KalmanFilter object.

        Parameters
        ----------
        F : np.ndarray
            State transition matrix of shape (n, n).
        H : np.ndarray
            Observation matrix of shape (m, n).
        Q : np.ndarray
            Process noise covariance matrix of shape (n, n).
        R : np.ndarray
            Measurement noise covariance matrix of shape (m, m).
        """
        if F.ndim != 2 or F.shape[0] != F.shape[1]:
            raise ValueError("F must be a square matrix of shape (n, n).")
        if H.ndim != 2 or H.shape[1] != F.shape[0]:
            raise ValueError("H must have shape (m, n) to match F.")
        if Q.shape != F.shape:
            raise ValueError("Q must have the same shape as F (n, n).")
        if R.ndim != 2 or R.shape[0] != R.shape[1] or R.shape[0] != H.shape[0]:
            raise ValueError("R must be a square matrix of shape (m, m).")
        self.F              = F
        self.H              = H
        self.Q              = Q
        self.R              = R
        self.state          = None  # initialised on first observation
        self.P              = None  # initialised on first observation
        self.num_iterations = 0
        self.innovation     = None

    def _init(self, observation: np.ndarray) -> None:
        """
        Bootstrap state and error covariance from the first observation.

        Parameters
        ----------
        observation : np.ndarray
            First observation vector of shape (m,).
        """
        self.state      = self.H.T @ observation
        self.P          = np.eye(self.F.shape[0])
        self.innovation = np.zeros(self.H.shape[0])

    def _predict(self) -> None:
        """
        Advance state and error covariance through the dynamics model.
        Mutates self.state and self.P in-place.
        """
        self.state = self.F @ self.state
        self.P     = self.F @ self.P @ self.F.T + self.Q

    def _correct(self, observation: np.ndarray) -> None:
        """
        Correct the predicted state with a new measurement.
        Mutates self.state, self.P, and self.innovation in-place.

        Parameters
        ----------
        observation : np.ndarray
            Observation vector of shape (m,).
        """
        innov_cov       = self.H @ self.P @ self.H.T + self.R
        self.K               = self.P @ self.H.T @ np.linalg.solve(
                              innov_cov, np.eye(self.R.shape[0])).T
        self.innovation = observation - self.H @ self.state
        self.state     += self.K @ self.innovation
        I_KH            = np.eye(self.P.shape[0]) - self.K @ self.H
        self.P          = I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T

    def update(self, observation: np.ndarray) -> None:
        """
        Update the model with the new observation.

        Parameters
        ----------
        observation : np.ndarray
            New observation vector of shape (m,).
        """
        if not isinstance(observation, np.ndarray):
            raise ValueError("Observation must be a numpy array.")
        if observation.ndim != 1:
            raise ValueError("Observation must be a 1D numpy array.")
        if observation.shape[0] != self.H.shape[0]:
            raise ValueError(
                f"Observation must have {self.H.shape[0]} elements to match H."
            )
        self.num_iterations += 1
        if self.state is None:  # first observation
            self._init(observation)
        else:                   # subsequent observations
            self._predict()
            self._correct(observation)

    def fit(self, observations: list[np.ndarray]) -> list[np.ndarray]:
        """
        Fit the filter to a set of observations.
        
        Parameters
        ----------
        observations : list of np.ndarray
            List of observation vectors, each of shape (m,).
        Returns
        -------
        list_smooth : list of np.ndarray
            List of filtered state estimates after each observation.
        """
        list_smooth = []
        for observation in observations:
            self.update(observation)
            list_smooth.append(self.state.copy())
        return list_smooth


if __name__ == "__main__":

    # Example usage of the filters

    # observations by sinusoidal function with noise
    t = np.linspace(0, 10, 100)
    observations = [np.array([np.sin(time) + np.random.normal(0, 0.1)]) for time in t] 

    # Recursive Average Filter
    recursive_filter = RecursiveAverage()
    recursive_means = recursive_filter.fit(observations)
    print("Recursive Average Filter Means:")
    for mean in recursive_means:
        print(mean)

    # Low-Pass Filter
    lowpass_filter = LowPassFilter(alpha=0.8)
    lowpass_means = lowpass_filter.fit(observations)
    print("\nLow-Pass Filter Means:")
    for mean in lowpass_means:
        print(mean)

    # Rolling Average Filter
    list_filtered = []
    rolling_filter = RollingAverageFilter(window=5)
    for observation in observations:
        rolling_filter.update(observation)
        list_filtered.append(rolling_filter.moving_mean)
        print(f"Rolling Average Filter Mean: {rolling_filter.moving_mean}")

    # Kalman Filter
    F = np.array([[1]])
    H = np.array([[1]])
    Q = np.array([[0.1]])
    R = np.array([[0.5]])
    kalman_filter = KalmanFilter(F, H, Q, R)
    kalman_states = kalman_filter.fit(observations)
    print("\nKalman Filter States:")
    for state in kalman_states:
        print(state)

    # Plotting the results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    plt.plot([obs[0] for obs in observations], label='Observations', marker='o')
    plt.plot([mean[0] for mean in recursive_means], label='Recursive Average Filter', marker='x')
    plt.plot([mean[0] for mean in lowpass_means], label='Low-Pass Filter', marker='s')
    plt.plot([mean[0] for mean in list_filtered], label='Rolling Average Filter', marker='d')
    plt.plot([state[0] for state in kalman_states], label='Kalman Filter', marker='^')
    plt.title('Filter Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()
