
import pytest
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from mlcf.windowing.filtering import LabelBalanceFilter
from mlcf.windowing.iterator.tseries_lite import TIME_INDEX_NAME


@pytest.mark.parametrize(
    "max_occ, window_width, window_step, k, expected",
    [
        (200, 30, 1, 1, 2400),
        (200, 30, 2, 1, 2400),
        (400, 30, 2, 1, 4364),
        (600, 100, 6, 1, 2542),
        (200, 30, 1, 2, 2400),
        (200, 30, 2, 2, 2400),
    ]
)
def test_label_balance_filter(ohlcvrl_btc, max_occ, window_width, window_step, k, expected):
    data = ohlcvrl_btc.copy()
    data[TIME_INDEX_NAME] = np.arange(len(data), dtype=int)
    window_filter = LabelBalanceFilter("label", max_occ, lambda li, k: li[:k], k)

    index_data = sliding_window_view(
        data[TIME_INDEX_NAME],
        window_shape=(window_width),
    ).reshape((-1, window_width))
    index_data = index_data[::window_step]
    index_data = index_data[window_filter(ohlcvrl_btc, index_data)]
    assert index_data.shape[0] == expected
