
import pytest
import numpy as np
from mlcf.datatools.standardisation import ClassicStd
from mlcf.windowing.iterator.forecast_iterator import WindowForecastIterator
from mlcf.windowing.iterator.tseries import WTSeries


@pytest.mark.parametrize(
    "args_wtseries, args_forecast",
    [
        (
            {
                "window_width": 5+1+0,
                "window_step": 1
            },
            {
                "input_width": 5,
                "target_width": 1,
                "input_features": None,
                "target_features": None,
                "std_by_feature": None,
                "index_selection_mode": None,
            }
        ),
    ]
)
def test_window_forecast_iterator(
    ohlcvr_btc,
    args_wtseries,
    args_forecast
):
    wtseries = WTSeries.create_wtseries(ohlcvr_btc.iloc[0:100], **args_wtseries)
    w_iterator = WindowForecastIterator(wtseries, **args_forecast)
    for w_fore, w_tseries in zip(w_iterator, wtseries):
        w_input, w_target = w_fore
        assert w_input.shape == (
            args_forecast["input_width"],
            len(list(
                args_forecast["input_features"]
                if args_forecast["input_features"]
                else wtseries.features))
        )
        assert w_target.shape == (
            args_forecast["target_width"],
            len(list(
                args_forecast["target_features"]
                if args_forecast["target_features"]
                else wtseries.features))
        )

        if args_forecast["index_selection_mode"]:
            pass
        else:
            assert np.all(w_tseries.iloc[:args_forecast["input_width"]].index == w_input.index)
            assert np.all(w_tseries.iloc[-args_forecast["target_width"]:].index == w_target.index)

        if args_forecast["std_by_feature"]:
            for feature, std_object in args_forecast["std_by_feature"].items():
                if isinstance(std_object, ClassicStd):
                    assert np.round(w_input[feature].mean(), 4) == 0.0
                    assert np.round(w_input[feature].std(), 4) == 1.0


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
