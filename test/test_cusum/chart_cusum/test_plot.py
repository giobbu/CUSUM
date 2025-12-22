import numpy as np

def test_cusum_plot_change_points(mocker, detector):
    """
    Test plot_change_points method of the ProbCUSUM detector.
    """

    data = np.array([1, 2, 3, 4, 5])
    upper_limits = np.array([0.8, 0.7, 0.6, 0.5, 0.4])
    lower_limits = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    cusums = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    change_points = [2]

    mock_fig = mocker.patch("source.detector.cusum.plt.figure")
    mock_subplt = mocker.patch("source.detector.cusum.plt.subplot")
    mock_plot = mocker.patch("source.detector.cusum.plt.plot")
    mock_axvline = mocker.patch("source.detector.cusum.plt.axvline")
    mock_xlabel = mocker.patch("source.detector.cusum.plt.xlabel")
    mock_ylabel = mocker.patch("source.detector.cusum.plt.ylabel")
    mock_title = mocker.patch("source.detector.cusum.plt.title")
    mock_legend = mocker.patch("source.detector.cusum.plt.legend")
    mock_grid = mocker.patch("source.detector.cusum.plt.grid")
    mock_layout = mocker.patch("source.detector.cusum.plt.tight_layout")
    mock_show = mocker.patch("source.detector.cusum.plt.show")


    detector.plot_change_points(
        data=data,
        change_points=change_points,
        upper_limits=upper_limits,
        lower_limits=lower_limits,
        cusums=cusums,
    )

    mock_plot.assert_any_call(
        data, color="blue", label="Data", linestyle="--"
    )

    mock_axvline.assert_called()
    mock_xlabel.assert_called()
    mock_ylabel.assert_called()
    mock_title.assert_called()
    mock_legend.assert_called()
    mock_grid.assert_called()
    mock_layout.assert_called()
    mock_show.assert_called_once()
