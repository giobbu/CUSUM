import numpy as np

def test_cusum_plot_change_points(mocker, detector):
    """Test plot_change_points method of the KS CUM detector."""

    data = np.array([1, 2, 1, 2, 1])
    change_points = [1, 3]
    p_values =  np.array([0.05, 0.01, 0.03, 0.02, 0.04]).reshape(-1, 1)

    # PATCH matplotlib NEL MODULO
    mock_figure = mocker.patch("source.detector.cusum.plt.figure")
    mock_subplot = mocker.patch("source.detector.cusum.plt.subplot")
    mock_plot = mocker.patch("source.detector.cusum.plt.plot")
    mock_axvline = mocker.patch("source.detector.cusum.plt.axvline")
    mock_axhline = mocker.patch("source.detector.cusum.plt.axhline")
    mock_legend = mocker.patch("source.detector.cusum.plt.legend")
    mock_tight = mocker.patch("source.detector.cusum.plt.tight_layout")
    mock_show = mocker.patch("source.detector.cusum.plt.show")
    mock_xlabel = mocker.patch("source.detector.cusum.plt.xlabel")
    mock_ylabel = mocker.patch("source.detector.cusum.plt.ylabel")
    mock_title = mocker.patch("source.detector.cusum.plt.title")

    detector.plot_change_points(
        data=data,
        change_points=change_points,
        p_values=p_values
    )

    assert mock_legend.call_count == 2
    assert mock_axvline.call_count == len(change_points)
    mock_tight.assert_called_once()
    mock_show.assert_called_once()

    mock_axvline.assert_called()
    mock_xlabel.assert_called()
    mock_ylabel.assert_called()
    mock_title.assert_called()
    mock_legend.assert_called()
    mock_tight.assert_called()
    mock_show.assert_called_once()

    