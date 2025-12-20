import numpy as np
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
        self.list_generators = [ChangePointGenerator(**params) if params is not None else ChangePointGenerator() for params in dict_streams]
        self.list_data_streams = []

    def generate_data_streams(self):
        """
        Generate data for all ChangePointGenerator instances and store the results.
        """
        for generator in self.list_generators:
            generator.generate_data()
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
        for _, generator in enumerate(self.list_generators):
            generator.plot_data()
