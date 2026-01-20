import numpy as np
import matplotlib.pyplot as plt
from source.generator.change_point_generator import ChangePointGenerator

class MultiDataStreams:
    """"
    Class to generate and manage multiple data streams with change points.

    Parameters
    ----------
    num_streams : int
        The number of data streams to generate.
    dict_streams : list
        A list of dictionaries, each containing parameters for a ChangePointGenerator.
    """
    def __init__(self, num_streams:int=2, dict_streams: list=[]):
        """
        Initialize ManyDataStreams with a list of ChangePointGenerator instances.

        Parameters
        ----------
        num_streams : int
            The number of data streams to generate.
        dict_streams : list
            A list of dictionaries, each containing parameters for a ChangePointGenerator.
        """
        if len(dict_streams) > 0:
            assert len(dict_streams) == num_streams, "Length of dict_streams must match num_streams."
        else:
            dict_streams = [None] * num_streams
        self.dict_streams = dict_streams
        self.list_generators = [ChangePointGenerator(**params) if params is not None else ChangePointGenerator() for params in self.dict_streams]
        self.list_data_streams = []
        self.dict_missing = None

    def generate_data_streams(self, dict_missing=None):
        """
        Generate data for all ChangePointGenerator instances and store the results.

        Parameters
        ----------
        dict_missing : list, optional
            A list of dictionaries specifying missing data parameters for each stream. Each dictionary can have the following keys:
            - 'type': 'point' or 'block'
            - 'percentage': float, percentage of data to be made missing
            - 'min_block_size': int, minimum size of blocks for block missingness (only for 'block' type)
            - 'max_block_size': int, maximum size of blocks for block missingness (only for 'block' type)
            If None, no missing data will be introduced.
        """
        self.dict_missing = dict_missing
        for i, generator in enumerate(self.list_generators):
            generator.generate_data()
            if self.dict_missing is not None:
                if self.dict_missing[i] is not None:
                    miss_params = self.dict_missing[i]
                    if miss_params['type'] == 'point':
                        data_stream = generator.generate_point_nans(miss_params['percentage'])
                    elif miss_params['type'] == 'block':
                        data_stream = generator.generate_block_nans(miss_params['percentage'], miss_params['min_block_size'], miss_params['max_block_size'])
                    else:
                        raise ValueError("Invalid missingness type. Use 'point' or 'block'.")
                else:
                    data_stream = generator.get_data()
                self.list_data_streams.append(data_stream)
            else:
                self.list_data_streams.append(generator.get_data())

    def get_all_streams(self):
        """
        Get the list of all generated data streams.

        Returns
        -------
        list
            A list of all generated data streams.
        """
        return self.list_data_streams
    
    def get_data_streams_as_array(self):
        """
        Get all generated data streams as a transposed NumPy array.
        
        Returns
        -------
        np.ndarray
            A transposed NumPy array of all generated data streams.
            Shape: (num_data_points, num_streams)
        """
        return np.array(self.list_data_streams).T
    
    def plot_all_streams(self):
        """
        Plot the data for all ChangePointGenerator instances.
        """
        fig, axes = plt.subplots(len(self.list_data_streams), 1, figsize=(20, 5 * len(self.list_data_streams)))
        for i, data_stream in enumerate(self.list_data_streams):
            axes[i].plot(data_stream, color='blue', label=f'Time Series Data Stream {i+1}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Value')
            axes[i].set_title(f'Generated Time Series Data Stream {i+1}')
            axes[i].legend()
            axes[i].grid(True)
        plt.tight_layout()
        plt.show()

            
            
