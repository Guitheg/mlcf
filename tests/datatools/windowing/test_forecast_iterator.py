
import pytest
from mlcf.datatools.windowing.forecast_iterator import WindowForecastIterator
from mlcf.datatools.windowing.tseries import WTSeries


@pytest.mark.parametrize(
    "input_width, target_width, offset, step, i_features, t_features",
    [
        (5, 1, 0, 1, None, None),
        (5, 3, 5, 2, None, None),
        (5, 3, 5, 10, None, None),
        (5, 3, 5, 1, None, None),
        (5, 3, 5, 2, ["close"], None),
        (5, 3, 5, 2, None, ["return"]),
        (5, 3, 5, 2, ["close"], ["return"]),
    ]
)
def test_window_forecast_iterator(
    ohlcvr_btc,
    input_width,
    target_width,
    offset,
    step,
    i_features,
    t_features
):
    wtseries = WTSeries.create_wtseries(
        ohlcvr_btc.iloc[0:100],
        window_width=input_width+target_width+offset,
        window_step=step
    )
    w_iterator = WindowForecastIterator(
        wtseries,
        input_width=input_width,
        target_width=target_width,
        input_features=i_features,
        target_features=t_features
    )
    for w in w_iterator:
        wi, wt = w
        assert len(wi) == input_width
        assert len(wt) == target_width
        assert list(wi.columns) == list(i_features if i_features else wtseries.features)
        assert list(wt.columns) == list(t_features if t_features else wtseries.features)


@pytest.mark.parametrize(
    "input_width, target_width, window_width, i_features, t_features, expected_exception",
    [
        (5, 3, 8, ["a"], None, AttributeError),
        (5, 3, 8, None, ["b"], AttributeError),
        (5, 3, 8, ["a"], ["b"], AttributeError),
        (5, 3, 7, None, None, ValueError),
    ]
)
def test_window_forecast_iterator_exception(
    ohlcvr_btc,
    input_width,
    target_width,
    window_width,
    i_features,
    t_features,
    expected_exception
):
    wtseries = WTSeries.create_wtseries(
        ohlcvr_btc.iloc[0:100],
        window_width=window_width,
        window_step=1
    )
    with pytest.raises(expected_exception):
        WindowForecastIterator(
            wtseries,
            input_width=input_width,
            target_width=target_width,
            input_features=i_features,
            target_features=t_features
        )
