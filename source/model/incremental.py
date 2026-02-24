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
        if observation.shape[0] != self.num_variables:
            raise ValueError(f"Observation must have {self.num_variables} variables.")
        if observation.ndim == 1:
            raise ValueError("Observation must be a column vector.")
        if len(label) != 1:
            raise ValueError("Label must be a single value.")
        if not isinstance(label, float):
            raise ValueError("Label must be a float.")
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
        if observation.shape[0] != self.num_variables:
            raise ValueError(f"Observation must have {self.num_variables} variables.")
        return (self.w.T @ observation)[0][0]
    



class KalmanFilter:
    """ Linear Kalman filter for state estimation. """
    def __init__(self, A : np.ndarray, H : np.ndarray, Q : np.ndarray, R : np.ndarray, x0 : np.ndarray, P0 : np.ndarray) -> None:
        self.A = A  # state transition matrix
        self.H = H  # observation matrix
        self.Q = Q  # state covariance
        self.R = R  # measurement covariance
        self.x = x0  # initial state
        self.P = P0  # initial state covariance

    def update(self, z : np.ndarray) -> None:
        " Update the state estimate with a new observation. "
        # Kalman filter update
        innovation = z - self.H @ self.x  # innovation
        innovation_cov = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(innovation_cov)  # Kalman gain
        self.x = self.x + K @ innovation  # state update
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P  # state covariance update

    def predict(self) -> tuple:
        " Predict the next state. "
        self.x = self.A @ self.x  # state prediction
        self.P = self.A @ self.P @ self.A.T + self.Q  # state covariance prediction
        return self.x, self.P


