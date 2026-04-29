import numpy as np
import matplotlib.pyplot as plt

class ChangePointGenerator:
    """
    A class to generate time series data with different types of change points.

    Parameters
    ----------
    num_segments : int
        Number of segments in the time series data.
    segment_length : int
        Length of each segment.
    change_point_type : str
        Type of change point to introduce ('sudden_shift', 'gradual_drift', 'periodic_change').
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, num_segments=3, segment_length=500, change_point_type='sudden_shift', seed=42):
        """
        Initializes the ChangePointGenerator with the specified parameters.

        Parameters
        ----------
        num_segments : int
            Number of segments in the time series data.
        segment_length : int
            Length of each segment.
        change_point_type : str
            Type of change point to introduce ('sudden_shift', 'gradual_drift', 'periodic_change').
        seed : int
            Random seed for reproducibility.
        """
        if not isinstance(num_segments, int) or num_segments <= 0:
            raise ValueError("num_segments must be a positive integer")
        if not isinstance(segment_length, int) or segment_length <= 0:
            raise ValueError("segment_length must be a positive integer")
        if change_point_type not in ['sudden_shift', 'gradual_drift', 'periodic_change']:
            raise ValueError("change_point_type must be one of: 'sudden_shift', 'gradual_drift', 'periodic_change'")
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")

        self.num_segments = num_segments
        self.segment_length = segment_length
        self.change_point_type = change_point_type
        self.seed = seed
        self.data = []

    def _add_sudden_shift(self):
        """
        Add a sudden shift change point to the data.

        Returns
        -------
        mean : float
            Mean value.
        std_dev : float
            Standard deviation value.
        """
        mean = np.random.uniform(0, 100)
        std_dev = np.random.uniform(5, 20)
        return mean, std_dev
    
    def _add_gradual_drift(self):
        """
        Add a gradual drift change point to the data.

        Returns
        -------
        mean : numpy array
            Mean values for the gradual drift.
        std_dev : float
            Standard deviation value.
        """
        mean_array = np.linspace(0, 50, self.segment_length)
        std_dev = np.random.uniform(5, 20)
        return mean_array, std_dev
    
    def _add_periodic_change(self):
        """
        Add a periodic change point to the data.

        Returns
        -------
        mean : numpy array
            Mean values for the periodic change.
        std_dev : float
            Standard deviation value.
        """
        mean_array = np.sin(np.linspace(0, 2 * np.pi, self.segment_length))
        std_dev = np.random.uniform(5, 20)
        return mean_array, std_dev
    
    
    
    def _segment_data(self, mean, std_dev):
        """
        Generate a segment of data based on the provided mean and standard deviation.

        Parameters
        ----------
        mean : float or numpy array
            Mean value(s) for the segment.
        std_dev : float
            Standard deviation value for the segment.

        Returns
        -------
        segment_data : numpy array
            Generated segment of data.
        """
        return np.random.normal(mean, std_dev, self.segment_length)


    def generate_data(self):
        """
        Generate time series data with different types of change points.
        """
        np.random.seed(self.seed)
        dict_shift = {
            'sudden_shift': self._add_sudden_shift,
            'gradual_drift': self._add_gradual_drift,
            'periodic_change': self._add_periodic_change
        }
        for _ in range(self.num_segments):
            mean, std_dev = dict_shift[self.change_point_type]()
            segment_data = self._segment_data(mean, std_dev)
            self.data.extend(segment_data)
        self.data = np.array(self.data)

    def get_data(self):
        """
        Returns the generated time series data.

        Returns
        -------
        data : numpy array
            The generated time series data.
        """
        return self.data

    def add_sudden_shift(self, mean_before, mean_after, std_dev_before, std_dev_after, change_point_index):
        """
        Add a sudden shift change point to the data.

        Parameters
        ----------
        mean_before : float
            Mean value before the change point.
        mean_after : float
            Mean value after the change point.
        std_dev_before : float
            Standard deviation before the change point.
        std_dev_after : float
            Standard deviation after the change point.
        change_point_index : int
            Index at which the change point occurs.
        """
        if not all(isinstance(val, (int, float)) for val in [mean_before, mean_after, std_dev_before, std_dev_after]):
            raise ValueError("mean_before, mean_after, std_dev_before, std_dev_after must be numbers")
        if not isinstance(change_point_index, int) or change_point_index < 0 or change_point_index >= len(self.data):
            raise ValueError("change_point_index must be a non-negative integer within the data range")

        self.data[change_point_index:] = np.random.normal(mean_after, std_dev_after, len(self.data) - change_point_index)
        self.data[:change_point_index] = np.random.normal(mean_before, std_dev_before, change_point_index)

    def add_gradual_drift(self, mean_start, mean_end, std_dev, change_point_index):
        """
        Add a gradual drift change point to the data.

        Parameters
        ----------
        mean_start : float
            Mean value at the start of the drift.
        mean_end : float
            Mean value at the end of the drift.
        std_dev : float
            Standard deviation during the drift.
        change_point_index : int
            Index at which the drift starts.
        """
        if not all(isinstance(val, (int, float)) for val in [mean_start, mean_end, std_dev]):
            raise ValueError("mean_start, mean_end, std_dev must be numbers")
        if not isinstance(change_point_index, int) or change_point_index < 0 or change_point_index >= len(self.data):
            raise ValueError("change_point_index must be a non-negative integer within the data range")

        drift_slope = (mean_end - mean_start) / (len(self.data) - change_point_index)
        for i in range(change_point_index, len(self.data)):
            self.data[i] = np.random.normal(mean_start + drift_slope * (i - change_point_index), std_dev)

    def add_periodic_change(self, amplitude, period, std_dev, change_point_index):
        """
        Add a periodic change point to the data.

        Parameters
        ----------
        amplitude : float
            Amplitude of the periodic change.
        period : float
            Period of the periodic change.
        std_dev : float
            Standard deviation during the periodic change.
        change_point_index : int
            Index at which the periodic change starts.
        """
        if not all(isinstance(val, (int, float)) for val in [amplitude, period, std_dev]):
            raise ValueError("amplitude, period, std_dev must be numbers")
        if not isinstance(change_point_index, int) or change_point_index < 0 or change_point_index >= len(self.data):
            raise ValueError("change_point_index must be a non-negative integer within the data range")

        for i in range(change_point_index, len(self.data)):
            self.data[i] = np.sin(2 * np.pi * (i - change_point_index) / period) * amplitude + np.random.normal(0, std_dev)

    def plot_data(self):
        """
        Plot the generated time series data.
        """
        plt.figure(figsize=(25, 5))
        plt.plot(self.data, color='blue', label='Time Series Data')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Generated Time Series Data')
        plt.legend()
        plt.grid(True)
        plt.show()

    def generate_point_nans(self, percentage):
        """
        Generate data with a specified percentage of NaN values.

        Parameters
        ----------
        percentage : float
            Percentage of NaN values to introduce in the data (between 0 and 1).
        
        Returns
        -------
        data_with_nans : numpy array
            Data with NaN values introduced.
        """
        if not 0 <= percentage <= 1:
            raise ValueError("nan_percentage must be between 0 and 1.")

        array_size = len(self.data)
        num_nan = int(array_size * percentage)
        nan_indices = np.random.choice(array_size, size=num_nan, replace=False)
        data_with_nans = np.copy(self.data)
        data_with_nans[nan_indices] = np.nan

        return data_with_nans
    
    def _get_num_blocks(self, percentage, min_block_size):
        """
        Calculate the number of blocks needed to achieve the specified percentage of NaN values.

        Parameters
        ----------
        percentage : float
            Percentage of NaN values to introduce in the data (between 0 and 1).
        min_block_size : int
            Minimum size of each block of NaNs.
        
        Returns
        -------
        num_elements : int
            Total number of elements in the data.
        num_blocks : int
            Number of blocks needed to achieve the specified percentage of NaN values.
        """
        if not 0 <= percentage <= 1:
            raise ValueError("percentage must be between 0 and 1.")
        if min_block_size <= 0:
            raise ValueError("min_block_size must be a positive integer.")
        num_elements = len(self.data)
        num_nan = int(num_elements * percentage)
        num_blocks = num_nan // min_block_size
        return num_elements, num_blocks
    
    def _get_start_indices_for_blocks(self, num_elements, num_blocks, min_block_size):
        """
        Get random start indices for blocks of NaN values.

        Parameters
        ----------
        num_elements : int
            Total number of elements in the data.
        num_blocks : int
            Number of blocks needed to achieve the specified percentage of NaN values.
        min_block_size : int
            Minimum size of each block of NaNs.

        Returns
        -------
        start_indices : numpy array
            Randomly selected start indices for the blocks of NaN values.
        """
        if num_blocks <= 0:
            raise ValueError("num_blocks must be a positive integer.")
        if min_block_size <= 0:
            raise ValueError("min_block_size must be a positive integer.")
        if num_elements < min_block_size:
            raise ValueError("num_elements must be greater than or equal to min_block_size.")
        start_indices = np.random.choice(num_elements - min_block_size + 1, num_blocks, replace=False)
        return start_indices
    
    def _get_block_nan(self, data_block_nans, list_start_indices, min_block_size, max_block_size):
        """
        Replace values with NaNs in blocks based on the provided start indices, block_min_size, and block_max_size.

        Parameters
        ----------
        data_block_nans : numpy array
            The data array to modify with NaN values.
        list_start_indices : numpy array
            Randomly selected start indices for the blocks of NaN values.
        min_block_size : int
            Minimum size of each block of NaNs.
        max_block_size : int
            Maximum size of each block of NaNs.

        Returns
        -------
        data_block_nans : numpy array
            The modified data array with NaN values in blocks.
        """
        for start_index in list_start_indices:
            block_size = np.random.randint(min_block_size, max_block_size + 1)
            end_index = start_index + block_size
            data_block_nans[start_index:end_index] = np.nan
        return data_block_nans

    def generate_block_nans(self, percentage, min_block_size, max_block_size):
        """
        Generate data with NaN values in contiguous blocks.
        
        Parameters
        ----------
        percentage : float
            Percentage of the data to be replaced with NaNs (between 0 and 1).
        min_block_size : int
            Minimum size of each block of NaNs.
        max_block_size : int
            Maximum size of each block of NaNs.
        """
        if percentage < 0 or percentage > 1:
            raise ValueError("Percentage should be between 0 and 1")
        if min_block_size <= 0 or max_block_size <= 0 or min_block_size > max_block_size:
            raise ValueError("Invalid block sizes")
        # Calculate the number of blocks needed to achieve the specified percentage of NaN values
        num_elements, num_blocks = self._get_num_blocks(percentage, min_block_size)
        # Get random start indices for blocks
        start_indices = self._get_start_indices_for_blocks(num_elements, num_blocks, min_block_size)
        # Create a copy of the original array to modify
        data_block_nans = np.copy(self.data)  # Make a copy to avoid modifying the original data
        # Replace values with NaNs in blocks
        data_block_nans = self._get_block_nan(data_block_nans, start_indices, min_block_size, max_block_size) 
        return data_block_nans

    def plot_data_with_nans(self, data_with_nans):
        """
        Plot the generated time series data with NaN values.

        Parameters
        ----------
        data_with_nans : numpy array
            Data with NaN values to plot.
        """
        plt.figure(figsize=(25, 5))
        plt.plot(data_with_nans, label='Time Series Data with NaNs')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Generated Time Series Data with NaNs')
        plt.legend()
        plt.grid(True)
        plt.show()