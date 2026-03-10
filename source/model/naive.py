import numpy as np

class Persistent:
    """Predictor that always predicts the last observed value.

    Parameters
    ----------
    num_variables : int
        Number of variables including the constant.
    """

    def __init__(self):
        """Initialize the predictor."""
        self.num_variables = 1
        self.last_observation = np.zeros((1, self.num_variables))
        self.num_observations = 0

    def update(self, observation):
        """
        Update the model with a new observation and label.

        Parameters
        ----------
        observation : numpy array
            Observation vector.
        """
        if observation is None:
            raise ValueError("Observation cannot be None.")
        if observation.shape[0] != 1:
            raise ValueError("Observation must have exactly one  observation.")
        if observation.shape[1] != self.num_variables:
            raise ValueError(f"Observation must have {self.num_variables} variables.")
        
        self.last_observation = np.array(observation)
        self.num_observations += 1

    def fit(self, observations):
        """
        Fit the model to a set of observations and labels.

        Parameters
        ----------
        observations : list of numpy arrays
            List of observation vectors.
        """
        if observations is None:
            raise ValueError("Observations cannot be None.")
        if not isinstance(observations, list):
            raise ValueError("Observations must be a list of numpy arrays.")
        if len(observations) == 0:
            raise ValueError("Observations list cannot be empty.")
        for i in range(len(observations)):
            observation = np.array(observations[i])
            self.update(observation)

    def predict(self):
        """
        Predict the next value based on the last observation.

        Returns
        -------
        float
            Predicted value for the next observation.
        """
        return self.last_observation
