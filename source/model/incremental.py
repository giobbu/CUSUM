import numpy as np

class RecursiveLeastSquares:
    """
    Recurrent Least Squares algorithm for online linear regression.

    Parameters
    ----------
    num_variables : int
        Number of variables including the constant.
    forgetting_factor : float
        Forgetting factor (lambda), usually close to 1.
    initial_delta : float
        Controls the initial state.
    """

    def __init__(self, num_variables, forgetting_factor, initial_delta):
        """
        Initialize the RecurrentLeastSquares object.

        Parameters
        ----------
        num_variables : int
            Number of variables including the constant.
        forgetting_factor : float
            Forgetting factor (lambda), usually close to 1.
        initial_delta : float
            Controls the initial state.
        """

        if num_variables <= 0:
            raise ValueError("Number of variables must be positive.")
        if forgetting_factor <= 0:
            raise ValueError("Forgetting factor must be positive.")
        if initial_delta <= 0:
            raise ValueError("Initial delta must be positive.")

        self.num_variables = num_variables
        self.A = initial_delta * np.identity(self.num_variables)
        self.w = np.zeros((self.num_variables, 1))
        self.forgetting_factor_inverse = 1 / forgetting_factor
        self.num_observations = 1
        self.residual = np.array([0.0]).reshape(1, -1)
        self.residual_sqr = np.array([0.0]).reshape(1, -1)
        self.rmse = np.array([0.0]).reshape(1, -1)

    def update(self, observation, label):
        """
        Update the model with a new observation and label.

        Parameters
        ----------
        observation : numpy array
            Observation vector.
        label : float
            True label corresponding to the observation.
        """
        z = self.forgetting_factor_inverse * self.A @ observation
        alpha = 1 / (1 + observation.T @ z)
        self.w += (label - alpha * observation.T @ (self.w + label * z)) * z
        self.A -= alpha * z @ z.T
        self.num_observations += 1
        self.residual = label - self.w.T @ observation
        self.residual_sqr = (label - self.w.T @ observation) ** 2

    def fit(self, observations, labels):
        """
        Fit the model to a set of observations and labels.
        
        Parameters
        ----------
        observations : list of numpy arrays
            List of observation vectors.
        labels : list of floats
            List of true labels corresponding to the observations.
        """
        if len(observations) != len(labels):
            raise ValueError("Number of observations must be equal to the number of labels.")

        for i in range(len(observations)):
            observation = np.transpose(np.matrix(observations[i]))
            self.update(observation, labels[i])

    def predict(self, observation):
        """
        Predict the value of a new observation.
        
        Parameters
        ----------
        observation : numpy array
            Observation vector.
        
        Returns
        -------
        prediction : float
            Predicted value for the observation.
        """
        return float(self.w.T @ observation)


class RecursiveAverage:
    """
    Recursive Average Filter for online mean calculation.

    Parameters
    ----------
    recursive_mean : np.ndarray, optional
        Initial recursive mean.
    num_iterations : int, optional
        Initial number of iterations.
    """
    def __init__(self, recursive_mean: np.ndarray = None, num_iterations: int = 0) -> None:
        """
        Initialize the RecursiveAverage object.
        
        Parameters
        ----------
        recursive_mean : np.ndarray, optional
            Initial recursive mean.
        num_iterations : int, optional
            Initial number of iterations.
        """
        self.recursive_mean = recursive_mean
        self.num_iterations = num_iterations

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
    lowpass_mean : np.ndarray, optional
        Initial low-pass mean.
    num_iterations : int, optional
        Initial number of iterations.
    alpha : float, optional
        Smoothing factor between 0 and 1.
    """
    def __init__(self, lowpass_mean: np.ndarray = None, num_iterations: int = 0, alpha: float = 0.999) -> None:
        """
        Initialize the LowPassFilter object.
        
        Parameters
        ----------
        lowpass_mean : np.ndarray, optional
            Initial low-pass mean.
        num_iterations : int, optional
            Initial number of iterations.
        alpha : float, optional
            Smoothing factor between 0 and 1.
        """
        self.lowpass_mean = lowpass_mean
        self.num_iterations = num_iterations
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
    