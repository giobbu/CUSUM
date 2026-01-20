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
        list_smooth = []
        for observation in observations:
            self.update(observation)
            list_smooth.append(self.lowpass_mean)
        return list_smooth