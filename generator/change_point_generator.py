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
        plt.figure(figsize=(20, 6))
        plt.plot(self.data, color='blue', label='Time Series Data')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Generated Time Series Data')
        plt.legend()
        plt.grid(True)
        plt.show()