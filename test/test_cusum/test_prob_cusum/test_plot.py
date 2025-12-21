import numpy as np

def test_cusum_plot_change_points(mocker, detector):
    """
    Test plot_change_points method of the ProbCUSUM detector.
    """

    data = np.array([1, 2, 3, 4, 5])
    probabilities = np.array([0.9, 0.8, 0.02, 0.9, 0.9])
    change_points = [2]

    mocker.patch("source.detector.cusum.plt.figure")
    mocker.patch("source.detector.cusum.plt.subplot")
    mock_plot = mocker.patch("source.detector.cusum.plt.plot")
    mock_axvline = mocker.patch("source.detector.cusum.plt.axvline")
    mock_axhline = mocker.patch("source.detector.cusum.plt.axhline")
    mock_contourf = mocker.patch("source.detector.cusum.plt.contourf")
    mocker.patch("source.detector.cusum.plt.xlabel")
    mocker.patch("source.detector.cusum.plt.ylabel")
    mocker.patch("source.detector.cusum.plt.title")
    mocker.patch("source.detector.cusum.plt.legend")
    mocker.patch("source.detector.cusum.plt.grid")
    mocker.patch("source.detector.cusum.plt.tight_layout")
    mocker.patch("source.detector.cusum.plt.show")


    detector.plot_change_points(
        data=data,
        change_points=change_points,
        probabilities=probabilities,
    )

    mock_plot.assert_any_call(
        data, color="blue", label="Data", linestyle="--"
    )

    mock_contourf.assert_called_once()
    mock_axvline.assert_called()
    mock_axhline.assert_called_once()