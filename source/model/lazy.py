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
    bandwidth : float
        Bandwidth for the KDE.
    memory_size : int
        Maximum number of observations to store in memory.
    """

    def __init__(self, alpha, k=5, decay="exponential", bandwidth = 1.0, memory_size=100):
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
        bandwidth : float
            Bandwidth for the KDE.
        memory_size : int
            Maximum number of observations to store in memory.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if decay not in ["exponential", "linear"]:
            raise ValueError("Unsupported decay type. Use 'exponential' or 'linear'.")
        if alpha <= 0:
            raise ValueError("Alpha must be a positive number.")
        if bandwidth <= 0:
            raise ValueError("Bandwidth must be a positive number.")
        if memory_size <= k:
            raise ValueError("Memory size must be greater than k.")
        
        self.alpha = alpha
        self.k = k
        self.decay = decay
        self.bandwidth = bandwidth
        self.memory_size = memory_size
        self.X = None
        self.y = None

    def _apply_exponential_decay(self, time_diff):
        """
        Apply exponential decay to weights based on time difference.

        Parameters
        ----------
        time_diff : array-like
            Time differences for each observation.

        Returns
        -------
        weights : numpy array
            Weights after applying exponential decay.
        """
        weights = np.exp(-self.alpha * time_diff)
        return weights
    
    def _apply_linear_decay(self, time_diff):
        """
        Apply linear decay to weights based on time difference.

        Parameters
        ----------
        time_diff : array-like
            Time differences for each observation.

        Returns
        -------
        weights : numpy array
            Weights after applying linear decay.
        """
        weights = 1 / (1 + self.alpha * time_diff)
        return weights
    
    def _apply_normalization(self, values):
        """
        Normalize values to sum to 1.

        Parameters
        ----------
        values : array-like
            values to be normalized.

        Returns
        -------
        norm_values : numpy array
            Normalized values.
        """
        norm_values = values / np.sum(values)
        return norm_values

    def _compute_weights(self, t_previous):
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
        decay_dict = {
            "exponential": self._apply_exponential_decay,
            "linear": self._apply_linear_decay
        }
        # Create time differences, the most recent observation has time_diff=1, the second most recent has time_diff=2, etc.
        time_diff = np.array([i for i in range(1, len(t_previous)+1)], dtype=float)[::-1]
        weights = decay_dict[self.decay](time_diff)
        norm_weights = self._apply_normalization(weights)
        return norm_weights
        
    def _apply_weight2distance(self, distances, weights):
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
        # example: the weighted distance should be high if if distance is LOW and weight is HIGH
        eps = 1e-5  # Small value to prevent division by zero
        weighted_distances = weights / (distances + eps)
        return weighted_distances

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
        if self.X is None:
            self.X = X_new
            self.y = y_new
        elif len(self.X) >= self.memory_size:
            self.X = np.vstack((self.X, X_new))
            self.y = np.hstack((self.y, y_new))
            self.X = self.X[-self.memory_size:]
            self.y = self.y[-self.memory_size:]
        else:
            self.X = np.vstack((self.X, X_new))
            self.y = np.hstack((self.y, y_new))

    def _apply_euclidean_distance(self, X, x_query):
        """
        Compute Euclidean distance between stored observations and a query point.

        Parameters
        ----------
        X : numpy array
            Stored observation vectors.
        x_query : numpy array
            Query observation vector.

        Returns
        -------
        distances : numpy array
            Euclidean distances from each stored observation to the query point.
        """
        distances = np.linalg.norm(X - x_query, axis=1)
        return distances

    def _apply_mean(self, knn_labels, knn_weights=None):
        """
        Aggregate neighbor labels using mean.

        Parameters
        ----------
        knn_labels : numpy array
            Labels of the k nearest neighbors.

        Returns
        -------
        prediction : float
            Mean of the neighbor labels.
        """
        y_eval = None
        prediction = np.mean(knn_labels)
        return {'y_eval': y_eval, 'kde': prediction}
    
    def _apply_weighted_mean(self, knn_labels, knn_weights):
        """
        Aggregate neighbor labels using weighted mean.

        Parameters
        ----------
        knn_labels : numpy array
            Labels of the k nearest neighbors.
        knn_weights : numpy array
            Weights for each neighbor.

        Returns
        -------
        prediction : float
            Weighted mean of the neighbor labels.
        """
        y_eval = None
        prediction = np.sum(knn_labels * knn_weights) / np.sum(knn_weights)
        return {'y_eval': y_eval, 'kde': prediction}
    
    def _apply_kde(self, knn_labels, knn_weights=None):
        """
        Aggregate neighbor labels using Kernel Density Estimation (KDE).

        Parameters
        ----------
        knn_labels : numpy array
            Labels of the k nearest neighbors.

        Returns
        -------
        kde_result : dict
            Dictionary containing 'y_eval' and 'kde' for the KDE result.
        """
        y_eval = np.linspace(np.min(self.y), np.max(self.y), 100)
        kde = self._KDE(y_eval, knn_labels, self.bandwidth)
        return {'y_eval': y_eval, 'kde': kde}
    
    def _select_top_k_neighbors(self, weighted_distances):
        """
        Select the indices of the top k nearest neighbors based on weighted distances.

        Parameters
        ----------
        weighted_distances : numpy array
            Weighted distances to the query point.

        Returns
        -------
        knn_indices : numpy array
            Indices of the top k nearest neighbors.
        """
        knn_indices = np.argsort(weighted_distances)[:self.k]
        return knn_indices
    
    def _trim_by_k_neighbors(self, array, knn_indices):
        """
        Trim an array to only include the top k nearest neighbors.

        Parameters
        ----------
        array : numpy array
            Array to be trimmed.
        knn_indices : numpy array
            Indices of the top k nearest neighbors.
        
        Returns
        -------
        trimmed_array : numpy array
            Array containing only the values corresponding to the top k nearest neighbors.
        """
        trimmed_array = array[knn_indices]
        return trimmed_array
    
    def lazy_predict(self, x_query, take="kde"):
        """
        Predict the value for a new observation using weighted KNN.

        Parameters
        ----------
        x_query : numpy array
            Query observation vector.
        take : str
            Method to aggregate neighbor labels ("mean" or "weighted_mean" or "kde").

        Returns
        -------
        prediction : float
            Predicted value for the query observation.
        """
        # aggregation methods once top-k neighbors are selected
        aggregation_dict = {
            "mean": self._apply_mean,
            "weighted_mean": self._apply_weighted_mean,
            "kde": self._apply_kde
        }
        if take not in list(aggregation_dict.keys()):
            raise ValueError("Unsupported aggregation method. Use 'mean', 'weighted_mean' or 'kde'.")
        
        distances = self._apply_euclidean_distance(self.X, x_query)  # Compute distances to query point
        norm_distances = self._apply_normalization(distances)  # Normalize distances
        weights = self._compute_weights(self.X)
        weighted_distances = self._apply_weight2distance(norm_distances, weights)
        knn_indices = self._select_top_k_neighbors(weighted_distances) 
        knn_labels = self._trim_by_k_neighbors(self.y, knn_indices)
        knn_weights = self._trim_by_k_neighbors(weights, knn_indices)
        prediction_dict = aggregation_dict[take](knn_labels, knn_weights) 
        return prediction_dict
        
    def _gaussian_kernel(self, x):
        """
        Gaussian kernel function.
        
        Parameters
        ----------
        x : numpy array
            Input values.
            
        Returns
        -------
        numpy array
            Gaussian kernel values."""
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

    def _KDE(self, y_eval, y_labels, bandwidth):
        """
        Kernel Density Estimation using Gaussian kernel.

        Parameters
        ----------
        y_eval : numpy array
            Points at which to evaluate the density.
        y_labels : numpy array
            Sample points from which to estimate the density.
        bandwidth : float
            Bandwidth for the kernel.
        
        Returns
        -------
        numpy array
            Estimated density at y_eval points.
        """
        n = len(y_labels)
        kde = np.zeros_like(y_eval, dtype=float)
        for i, yi in enumerate(y_eval):
            dist = (yi - y_labels) / bandwidth
            kde[i] = np.sum(self._gaussian_kernel(dist))
        return kde/(n * bandwidth)


            







            

        


