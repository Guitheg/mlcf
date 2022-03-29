
import pytest
import numpy as np
from mlcf.datatools.standardisation import ClassicStd, MinMaxStd
from mlcf.datatools.utils import subset_selection
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
        (
            {
                "window_width": 5+1+0,
                "window_step": 1
            },
            {
                "input_width": 5,
                "target_width": 1,
                "input_features": ["close", "open"],
                "target_features": ["return"],
                "std_by_feature": None,
                "index_selection_mode": None,
            }
        ),
        (
            {
                "window_width": 5+1+5,
                "window_step": 1
            },
            {
                "input_width": 5,
                "target_width": 1,
                "input_features": ["close", "open"],
                "target_features": ["return"],
                "std_by_feature": None,
                "index_selection_mode": "default",
            }
        ),
        (
            {
                "window_width": 5+2+5,
                "window_step": 1
            },
            {
                "input_width": 5,
                "target_width": 2,
                "input_features": ["close", "open"],
                "target_features": ["return"],
                "std_by_feature": None,
                "index_selection_mode": "default",
            }
        ),
        (
            {
                "window_width": 5+2+5,
                "window_step": 1
            },
            {
                "input_width": 5,
                "target_width": 2,
                "input_features": ["close", "open"],
                "target_features": ["return"],
                "std_by_feature": None,
                "index_selection_mode": [2, -3, 3, -2, 2],
            }
        ),
        (
            {
                "window_width": 50+10,
                "window_step": 1
            },
            {
                "input_width": 50,
                "target_width": 10,
                "input_features": ["close", "volume", "return"],
                "target_features": ["return", "volume", "close"],
                "std_by_feature":
                    {
                        "close": ClassicStd(),
                        "return": ClassicStd(with_mean=False),
                        "volume": MinMaxStd((1, 100))
                    },
                "index_selection_mode": [0],
            }
        ),
        (
            {
                "window_width": 50+10+10,
                "window_step": 1
            },
            {
                "input_width": 50,
                "target_width": 10,
                "input_features": ["close", "volume", "return"],
                "target_features": ["return", "volume", "close"],
                "std_by_feature":
                    {
                        "close": ClassicStd(),
                        "return": ClassicStd(with_mean=False),
                        "volume": MinMaxStd((1, 100))
                    },
                "index_selection_mode": [-5, 50, -5, 10],
            }
        ),
        (
            {
                "window_width": 50+10+10,
                "window_step": 1
            },
            {
                "input_width": 50,
                "target_width": 10,
                "input_features": ["close", "volume", "return"],
                "target_features": ["return", "volume", "close"],
                "std_by_feature":
                    {
                        "close": ClassicStd(),
                        "return": ClassicStd(with_mean=False),
                        "volume": MinMaxStd((1, 100))
                    },
                "index_selection_mode": [-5, 50, -5, 10]
            }
        ),
    ]
)
def test_window_forecast_iterator(
    ohlcvr_btc,
    args_wtseries,
    args_forecast
):
    wtseries = WTSeries.create_wtseries(ohlcvr_btc.iloc[:70], **args_wtseries)
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
            selected_index = subset_selection(
                list(np.arange(wtseries.width)),
                w_iterator.index_selection_mode)
            input_index = selected_index[:args_forecast["input_width"]]
            target_index = selected_index[-args_forecast["target_width"]:]
            assert np.all(w_tseries.iloc[input_index].index == w_input.index)
            assert np.all(w_tseries.iloc[target_index].index == w_target.index)
        else:
            assert np.all(w_tseries.iloc[:args_forecast["input_width"]].index == w_input.index)
            assert np.all(w_tseries.iloc[-args_forecast["target_width"]:].index == w_target.index)

        if args_forecast["std_by_feature"]:
            for feature, std_object in args_forecast["std_by_feature"].items():
                if isinstance(std_object, ClassicStd):
                    if std_object.kwargs["with_mean"]:
                        assert np.round(w_input[feature].mean(), 4) == 0.0
                    if std_object.kwargs["with_std"]:
                        assert np.round(w_input[feature].std(), 1) == 1.0
                if isinstance(std_object, MinMaxStd):
                    assert min(
                        [
                            np.round(w_input[feature].min(), 2),
                            np.round(w_target[feature].min(), 2)
                        ]
                    ) >= np.round(w_tseries[feature].min() / (
                        std_object.minmax[1] - std_object.minmax[0]), 2)
                    assert max(
                        [
                            np.round(w_input[feature].max(), 2),
                            np.round(w_target[feature].max(), 2)
                        ]
                    ) >= np.round(w_tseries[feature].max() / (
                        std_object.minmax[1] - std_object.minmax[0]), 2)


@pytest.mark.parametrize(
    "args_wtseries, args_forecast, expected_exception",
    [
        (
            {
                "window_width": 50+10+10,
                "window_step": 1
            },
            {
                "input_width": 50,
                "target_width": 10,
                "input_features": ["close", "volume", "return", "adx"],
                "target_features": ["return", "volume", "close"],
                "std_by_feature":
                    {
                        "close": ClassicStd(),
                        "return": ClassicStd(with_mean=False),
                        "volume": MinMaxStd((1, 100))
                    },
                "index_selection_mode": [-5, 50, -5, 10],
            },
            AttributeError
        ),
        (
            {
                "window_width": 50+10+10,
                "window_step": 1
            },
            {
                "input_width": 50,
                "target_width": 10,
                "input_features": ["close", "volume", "return"],
                "target_features": ["return", "volume", "close", "adx"],
                "std_by_feature":
                    {
                        "close": ClassicStd(),
                        "return": ClassicStd(with_mean=False),
                        "volume": MinMaxStd((1, 100))
                    },
                "index_selection_mode": [-5, 50, -5, 10],
            },
            AttributeError
        ),
        (
            {
                "window_width": 40,
                "window_step": 1
            },
            {
                "input_width": 50,
                "target_width": 10,
                "input_features": ["close", "volume", "return"],
                "target_features": ["return", "volume", "close"],
                "std_by_feature":
                    {
                        "close": ClassicStd(),
                        "return": ClassicStd(with_mean=False),
                        "volume": MinMaxStd((1, 100))
                    },
                "index_selection_mode": [-5, 50, -5, 10],
            },
            ValueError
        ),
        (
            {
                "window_width": 60,
                "window_step": 1
            },
            {
                "input_width": 50,
                "target_width": 10,
                "input_features": ["close", "volume", "return"],
                "target_features": ["return", "volume", "close"],
                "std_by_feature":
                    {
                        "close": ClassicStd(),
                        "return": ClassicStd(with_mean=False),
                        "volume": MinMaxStd((1, 100))
                    },
                "index_selection_mode": [-5, 50, -5, -10, 5],
            },
            ValueError
        ),
        (
            {
                "window_width": 60,
                "window_step": 1
            },
            {
                "input_width": 50,
                "target_width": 10,
                "input_features": ["close", "volume", "return"],
                "target_features": ["return", "volume", "close"],
                "std_by_feature":
                    {
                        "close": ClassicStd(),
                        "return": ClassicStd(with_mean=False),
                        "volume": MinMaxStd((1, 100))
                    },
                "index_selection_mode": [-5, 5],
            },
            ValueError
        ),
    ]
)
def test_window_forecast_iterator_exception(
    ohlcvr_btc,
    args_wtseries,
    args_forecast,
    expected_exception
):
    wtseries = WTSeries.create_wtseries(
        ohlcvr_btc.iloc[:200],
        **args_wtseries
    )
    with pytest.raises(expected_exception):
        WindowForecastIterator(
            wtseries,
            **args_forecast
        )
