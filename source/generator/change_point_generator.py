import numpy as np
import matplotlib.pyplot as plt

class ChangePointGenerator:
    """
        A class to generate time series data with different types of change points.

        Example:
        ```
        # Example usage
        generator = ChangePointGenerator(num_segments=2, segment_length=1000, change_point_type='gradual_drift')
        generator.generate_data()
        generator.add_gradual_drift(10, 50, 5, 800)
        generator.plot_data()
        ```
    """

    def __init__(self, num_segments=3, segment_length=500, change_point_type='sudden_shift'):
        """
        Initializes the ChangePointGenerator with the specified parameters.
        """
        if not isinstance(num_segments, int) or num_segments <= 0:
            raise ValueError("num_segments must be a positive integer")
        if not isinstance(segment_length, int) or segment_length <= 0:
            raise ValueError("segment_length must be a positive integer")
        if change_point_type not in ['sudden_shift', 'gradual_drift', 'periodic_change']:
            raise ValueError("change_point_type must be one of: 'sudden_shift', 'gradual_drift', 'periodic_change'")

        self.num_segments = num_segments
        self.segment_length = segment_length
        self.change_point_type = change_point_type
        self.data = []

    def generate_data(self):
        """
        Generate time series data with different types of change points.
        """
        for _ in range(self.num_segments):
            if self.change_point_type == 'sudden_shift':
                mean = np.random.uniform(0, 100)
                std_dev = np.random.uniform(5, 20)
            elif self.change_point_type == 'gradual_drift':
                mean = np.linspace(0, 50, self.segment_length)
                std_dev = np.random.uniform(5, 20)
            elif self.change_point_type == 'periodic_change':
                mean = np.sin(np.linspace(0, 2 * np.pi, self.segment_length))
                std_dev = np.random.uniform(5, 20)
            segment_data = np.random.normal(mean, std_dev, self.segment_length)
            self.data.extend(segment_data)

    def add_sudden_shift(self, mean_before, mean_after, std_dev_before, std_dev_after, change_point_index):
        """
        Add a sudden shift change point to the data.
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

    def generate_random_nans(self, nan_percentage):
        """
        Generate data with a specified percentage of NaN values.

        Parameters:
        - nan_percentage: float, percentage of NaN values desired in the data

        Returns:
        - data_with_nans: numpy array, data with NaN values
        """

        if not 0 <= nan_percentage <= 1:
            raise ValueError("nan_percentage must be between 0 and 1.")

        array_size = len(self.data)
        num_nan = int(array_size * nan_percentage)

        nan_indices = np.random.choice(array_size, size=num_nan, replace=False)
        data_with_nans = np.copy(self.data)  # Make a copy to avoid modifying the original data
        data_with_nans[nan_indices] = np.nan

        return data_with_nans

    def generate_no_random_nans(self, percentage, min_block_size, max_block_size):
        
        """
        Replaces a percentage of values in a NumPy array with NaNs
        arranged in blocks of consecutive NaNs.

        Parameters:
        - array (numpy.ndarray): The input NumPy array.
        - percentage (float): The percentage of values to replace with NaNs.
                                Should be between 0 and 1.
        - min_block_size (int): The minimum size of each block of consecutive NaNs.
        - max_block_size (int): The maximum size of each block of consecutive NaNs.

        Returns:
        - data_with_nans (numpy.ndarray): The array with NaNs replacing
                                            the specified percentage of values
                                            in blocks of consecutive NaNs.
        """
        if percentage < 0 or percentage > 1:
            raise ValueError("Percentage should be between 0 and 1")

        if min_block_size <= 0 or max_block_size <= 0 or min_block_size > max_block_size:
            raise ValueError("Invalid block sizes")

        # Calculate the number of elements to replace with NaNs
        num_elements = len(self.data)
        num_nans = int(num_elements * percentage)

        # Calculate the number of blocks
        num_blocks = num_nans // min_block_size

        # Get random start indices for blocks
        start_indices = np.random.choice(num_elements - min_block_size + 1, num_blocks, replace=False)

        # Create a copy of the original array to modify
        data_with_nans = np.copy(self.data)  # Make a copy to avoid modifying the original data

        # Replace values with NaNs in blocks
        for start_index in start_indices:
            # Randomly select block size
            block_size = np.random.randint(min_block_size, max_block_size + 1)
            end_index = start_index + block_size
            data_with_nans[start_index:end_index] = np.nan
            
        return data_with_nans

    def plot_data_with_nans(self, data_with_nans):
        """
        Plot the generated time series data with NaN values.

        Parameters:
        - data_with_nans: numpy array, data with NaN values
        """
        plt.figure(figsize=(25, 5))
        plt.plot(data_with_nans, color='red', label='Time Series Data with NaNs')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Generated Time Series Data with NaNs')
        plt.legend()
        plt.grid(True)
        plt.show()