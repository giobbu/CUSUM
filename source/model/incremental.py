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


