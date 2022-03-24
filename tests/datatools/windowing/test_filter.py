
import pytest
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from mlcf.datatools.windowing.filter import LabelBalanceFilter
from mlcf.datatools.windowing.tseries import TIME_INDEX_NAME


@pytest.mark.parametrize(
    "max_count, window_width, window_step, expected",
    [
        (200, 30, 1, 2400),
        (200, 30, 2, 2400),
        (400, 30, 2, 4364),
        (600, 100, 6, 2542),
    ]
)
def test_label_balance_filter(ohlcvrl_btc, max_count, window_width, window_step, expected):
    data = ohlcvrl_btc.copy()
    data[TIME_INDEX_NAME] = np.arange(len(data), dtype=int)
    window_filter = LabelBalanceFilter("label", max_count, lambda li, k: li[:k])

    index_data = sliding_window_view(
        data[TIME_INDEX_NAME],
        window_shape=(window_width),
    ).reshape((-1, window_width))
    index_data = index_data[::window_step]
    window_filter(ohlcvrl_btc, index_data)
    index_data = index_data[[window_filter[idx] for idx in index_data]]
    assert index_data.shape[0] == expected
