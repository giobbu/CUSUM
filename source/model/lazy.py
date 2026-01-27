import numpy as np

class WeightedKNN:
    """
    Weighted K-Nearest Neighbors with exponential decay weighting.

    Parameters
    ----------
    alpha : float
        Decay rate for the weights.
    k : int
        Number of nearest neighbors to consider.
    decay : str
        Type of decay to use ("exponential", "linear" supported).
    """

    def __init__(self, alpha, k=5, decay="exponential"):
        """
        Initialize the WeightedKNN object.
        
        Parameters
        ----------
        alpha : float
            Decay rate for the weights.
        k : int
            Number of nearest neighbors to consider.
        decay : str
            Type of decay to use ("exponential", "linear" supported).
        """
        self.alpha = alpha
        self.k = k
        self.decay = decay

    def compute_weights(self, t_previous):
        """
        Compute weights based on time decay.
        
        Parameters
        ----------
        t_previous : array-like
            Previous time steps or indices.

        Returns
        -------
        norm_weights : numpy array
            Normalized weights for each previous observation.
        """
        time_diff = np.array([i for i in range(1, len(t_previous)+1)], dtype=float)[::-1]
        if self.decay == "linear":
            weights = 1 / (1 + self.alpha * time_diff)
        elif self.decay == "exponential":
            weights = np.exp(-self.alpha * time_diff)
        else:
            raise ValueError("Unsupported decay type.")
        norm_weights = weights / np.sum(weights)
        return norm_weights
        
    def apply_weight_dist(self, distances, weights):
        """
        Apply weights to distances.
        
        Parameters
        ----------
        distances : numpy array
            Distances to the query point.
        weights : numpy array
            Weights for each observation.

        Returns
        -------
        weighted_distances : numpy array
            Weighted distances.
        """
        weighted_distances = distances/weights
        return weighted_distances

    def init_store(self, X, y):
        """
        Initialize the data store with initial observations and labels.
        
        Parameters
        ----------
        X : numpy array
            Initial observation vectors.
        y : numpy array
            Initial labels corresponding to the observations.
        """
        self.X = X
        self.y = y

    def update_store(self, X_new, y_new):
        """
        Update the data store with new observations and labels.

        Parameters
        ----------
        X_new : numpy array
            New observation vectors.
        y_new : numpy array
            New labels corresponding to the observations.
        """
        self.X = np.vstack((self.X[1:], X_new))
        self.y = np.hstack((self.y[1:], y_new))

    def lazy_predict(self, x_query, take="mean"):
        """
        Predict the value for a new observation using weighted KNN.

        Parameters
        ----------
        x_query : numpy array
            Query observation vector.
        take : str
            Method to aggregate neighbor labels ("mean" or "weighted_mean").

        Returns
        -------
        prediction : float
            Predicted value for the query observation.
        """
        distances = np.linalg.norm(self.X - x_query, axis=1)
        norm_distances = distances / np.sum(distances)
        weights = self.compute_weights(self.X)
        weighted_distances = self.apply_weight_dist(norm_distances, weights)
        knn_indices = np.argsort(weighted_distances)[:self.k] 
        knn_labels = self.y[knn_indices]
        if take == "mean":
            prediction = np.mean(knn_labels)
            return prediction
        elif take == "weighted_mean":
            knn_weights = weights[knn_indices]
            prediction = np.sum(knn_labels * knn_weights) / np.sum(knn_weights)
            return prediction
        else:
            raise ValueError("Unsupported aggregation method.")


