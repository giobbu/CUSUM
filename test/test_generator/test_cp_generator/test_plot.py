import numpy as np


# def plot_data(self):
#     """
#     Plot the generated time series data.
#     """
#     plt.figure(figsize=(25, 5))
#     plt.plot(self.data, color='blue', label='Time Series Data')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.title('Generated Time Series Data')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def test_generator_plot(mocker, generator):
    """
    Test plot_data method of the ChangePointGenerator. 
    """
    
    mock_figure = mocker.patch("source.generator.change_point_generator.plt.figure")
    mock_plot = mocker.patch("source.generator.change_point_generator.plt.plot")
    mock_xlabel = mocker.patch("source.generator.change_point_generator.plt.xlabel")
    mock_ylabel = mocker.patch("source.generator.change_point_generator.plt.ylabel")
    mock_title = mocker.patch("source.generator.change_point_generator.plt.title")
    mock_legend = mocker.patch("source.generator.change_point_generator.plt.legend")
    mock_grid = mocker.patch("source.generator.change_point_generator.plt.grid")
    mock_show = mocker.patch("source.generator.change_point_generator.plt.show")

    generator.plot_data()

    mock_figure.assert_called_once_with(figsize=(25, 5))
    mock_plot.assert_called_once_with(generator.data, color='blue', label='Time Series Data')
    mock_xlabel.assert_called_once_with('Time')
    mock_ylabel.assert_called_once_with('Value')
    mock_title.assert_called_once_with('Generated Time Series Data')
    mock_legend.assert_called_once()
    mock_grid.assert_called_once_with(True)
    mock_show.assert_called_once()
   