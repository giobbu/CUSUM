import numpy as np

def test_cusum_plot_change_points(mocker, detector):
    """Test plot_change_points method of the CUSUM detector."""

    data = np.array([1, 2, 3, 2, 1])
    change_points = [2, 4]
    pos_changes = [0, 1, 2, 1, 0]
    neg_changes = [0, -1, -2, -1, 0]

    # PATCH matplotlib NEL MODULO
    mock_figure = mocker.patch("source.detector.cusum.plt.figure")
    mock_subplot = mocker.patch("source.detector.cusum.plt.subplot")
    mock_plot = mocker.patch("source.detector.cusum.plt.plot")
    mock_axvline = mocker.patch("source.detector.cusum.plt.axvline")
    mock_axhline = mocker.patch("source.detector.cusum.plt.axhline")
    mock_legend = mocker.patch("source.detector.cusum.plt.legend")
    mock_tight = mocker.patch("source.detector.cusum.plt.tight_layout")
    mock_show = mocker.patch("source.detector.cusum.plt.show")

    detector.plot_change_points(
        data=data,
        change_points=change_points,
        pos_changes=pos_changes,
        neg_changes=neg_changes,
    )

    assert mock_subplot.call_count == 2
    mock_plot.assert_any_call(
        data, color="blue", label="Data", linestyle="--"
    )

    assert mock_axvline.call_count == len(change_points)

    mock_axhline.assert_called_once_with(
        detector.threshold,
        color="red",
        linestyle="dashed",
        lw=2,
    )
    assert mock_legend.call_count == 2
    mock_tight.assert_called_once()
    mock_show.assert_called_once()