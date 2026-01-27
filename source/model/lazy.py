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
    """

    def __init__(self, alpha, k=5, decay="exponential", bandwidth = 1.0):
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
        """
        self.alpha = alpha
        self.k = k
        self.decay = decay
        self.bandwidth = bandwidth

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
        elif take == "kde":
            y_eval = np.linspace(np.min(self.y), np.max(self.y), 100)
            
            kde = self.KDE(y_eval, knn_labels, self.bandwidth)
            return {'y_eval': y_eval, 'kde': kde}
        else:
            raise ValueError("Unsupported aggregation method.")
        
    def gaussian_kernel(self, x):
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

    def KDE(self, y_eval, y_labels, bandwidth):
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
            kde[i] = np.sum(self.gaussian_kernel(dist))
        return kde/(n * bandwidth)
    
    
if __name__ == "__main__":
    
    # example with lazy predict
    wknn = WeightedKNN(alpha=0.5, k=3, decay="exponential")
    n = 1000

    # Create time series components
    t = np.arange(n)

    # Trend component (slowly increasing)
    trend = 0.05 * t

    # Seasonal component (yearly pattern)
    seasonal = 10 * np.sin(3 * np.pi * t / 365)

    # Random noise
    noise = np.random.normal(0, 2, n)

    # Combine components
    values = 50 + trend + seasonal + noise

    values[300:500] += 20  # Introduce a change point

    values[700:800] -= 15  # Introduce another change point

    n_lags = 3
    store_window = 100
    X_lags = np.zeros((store_window, n_lags))
    y = np.zeros((store_window, ))


    model = WeightedKNN(alpha=0.5, k=10, decay="exponential", bandwidth=2.0)

    # Store predictions and observed values
    list_predictions_mean = []
    list_predictions_weighted_mean = []
    list_observed = []

    for i in range(n_lags, n):
        if i < store_window+n_lags:
            continue
        elif i == store_window+n_lags:
            X = values[:store_window+n_lags].reshape(-1, 1)
            for j in range(n_lags):
                X_lags[:, j] = X[j:X.shape[0]-n_lags+j, 0]
            y = values[n_lags:store_window+n_lags]
            model.init_store(X_lags, y)
        elif i < n-1:
            x_query = np.stack([values[i], values[i-1], values[i-2]]).reshape(1, -1)
            prediction = model.lazy_predict(x_query, take="mean")
            list_predictions_mean.append(prediction)
            result = model.lazy_predict(x_query, take="kde")
            kde = result['kde']
            y_eval = result['y_eval']

            mean = np.sum(y_eval * kde) / np.sum(kde)
            q10 = y_eval[np.searchsorted(np.cumsum(kde)/np.sum(kde), 0.1)]
            q90 = y_eval[np.searchsorted(np.cumsum(kde)/np.sum(kde), 0.9)]
            list_predictions_weighted_mean.append(mean)
            list_observed.append(values[i+1])

            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.plot(y_eval, kde, label='KDE')
            plt.axvline(x=mean, color='g', linestyle='-', label='Weighted Mean')
            plt.axvline(x=q10, color='b', linestyle='--', label='10th Percentile')
            plt.axvline(x=q90, color='b', linestyle='--', label='90th Percentile')
            plt.axvline(x=values[i+1], color='r', linestyle='--', label='Observed Value')
            plt.title('fKDE of Predicted Values with Observed Value at step {}'.format(i+1))
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.show()

            model.update_store(x_query, np.array([values[i+1]]))
        


