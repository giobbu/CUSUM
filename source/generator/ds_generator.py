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

    def __str__(self):
        """
        Return a string representation of the MultiDataStreams instance.

        Returns
        -------
        str
            A string describing the MultiDataStreams instance.
        """
        return f"MultiDataStreams(num_streams={len(self.list_generators)}, dict_streams={self.dict_streams})"

    def _add_point_missingness(self, generator, percentage):
        """
        Add point missingness to the data streams.

        Parameters
        ----------
        generator : ChangePointGenerator
            The ChangePointGenerator instance to modify.
        percentage : float
            The percentage of data points to be made missing.
        """
        return generator.generate_point_nans(percentage)
    
    def _add_block_missingness(self, generator, percentage, min_block_size, max_block_size):
        """
        Add block missingness to the data streams.

        Parameters
        ----------
        generator : ChangePointGenerator
            The ChangePointGenerator instance to modify.
        percentage : float
            The percentage of data points to be made missing.
        min_block_size : int
            The minimum size of blocks for block missingness.
        max_block_size : int
            The maximum size of blocks for block missingness.
        """
        return generator.generate_block_nans(percentage, min_block_size, max_block_size)
    
    def _get_missing_dict_stream(self, index):
        """
        Get the missing data dictionary for a specific stream.

        Parameters
        ----------
        index : int
            The index of the stream for which to retrieve the missing data dictionary.

        Returns
        -------
        dict or None
            The missing data dictionary for the specified stream, or None if not set.
        """
        if self.dict_missing is not None:
            return self.dict_missing[index]
        return None


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
        for i, generator in enumerate(self.list_generators):
            generator.generate_data()
            dict_missing_stream = self._get_missing_dict_stream(i)
            if dict_missing_stream is not None:
                if dict_missing_stream['type'] == 'point':
                    data_stream = self._add_point_missingness(generator,
                                                              dict_missing_stream['percentage'])
                elif dict_missing_stream['type'] == 'block':
                    data_stream = self._add_block_missingness(generator, 
                                                              dict_missing_stream['percentage'],
                                                              dict_missing_stream['min_block_size'],
                                                              dict_missing_stream['max_block_size']
                                                              )
                else:
                    raise ValueError("Invalid missingness type. Use 'point' or 'block'.")
            else:
                data_stream = generator.get_data()
            self.list_data_streams.append(data_stream)

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

if __name__ == "__main__":
    mds = MultiDataStreams(num_streams=2, dict_streams=[{"seed": 42}, {"seed": 43}])
    mds.generate_data_streams(dict_missing=[{"type": "point", "percentage": 0.1}, {"type": "block", "percentage": 0.2, "min_block_size": 5, "max_block_size": 10}])
    mds.plot_all_streams()

            
            
