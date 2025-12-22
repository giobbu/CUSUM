import numpy as np

def test_cusum_plot_change_points(mocker, detector):
    """
    Test plot_change_points method of the ProbCUSUM detector.
    """

    data = np.array([1, 2, 3, 4, 5])
    probabilities = np.array([0.9, 0.8, 0.02, 0.9, 0.9])
    change_points = [2, 3]

    mock_figure = mocker.patch("source.detector.cusum.plt.figure")
    mock_subplt = mocker.patch("source.detector.cusum.plt.subplot")
    mock_plot = mocker.patch("source.detector.cusum.plt.plot")
    mock_axvline = mocker.patch("source.detector.cusum.plt.axvline")
    mock_axhline = mocker.patch("source.detector.cusum.plt.axhline")
    mock_contourf = mocker.patch("source.detector.cusum.plt.contourf")
    mock_xlabel = mocker.patch("source.detector.cusum.plt.xlabel")
    mock_ylabel = mocker.patch("source.detector.cusum.plt.ylabel")
    mock_title = mocker.patch("source.detector.cusum.plt.title")
    mock_legend = mocker.patch("source.detector.cusum.plt.legend")
    mock_grid = mocker.patch("source.detector.cusum.plt.grid")
    mock_tight = mocker.patch("source.detector.cusum.plt.tight_layout")
    mock_show = mocker.patch("source.detector.cusum.plt.show")


    detector.plot_change_points(
        data=data,
        change_points=change_points,
        probabilities=probabilities,
    )

    mock_plot.assert_any_call(
        data, color="blue", label="Data", linestyle="--"
    )

    assert mock_legend.call_count == 2
    assert mock_axvline.call_count == len(change_points)*2

    mock_contourf.assert_called_once()
    mock_axvline.assert_called()
    mock_axhline.assert_called_once()
    mock_xlabel.assert_called()
    mock_ylabel.assert_called()
    mock_title.assert_called()
    mock_legend.assert_called()
    mock_grid.assert_called()
    mock_tight.assert_called()
    mock_show.assert_called_once()