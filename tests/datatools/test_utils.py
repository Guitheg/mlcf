import pytest
import pandas as pd
import numpy as np
from mlcf.datatools.utils import split_pandas, to_train_val_test, \
                                 split_in_interval, input_target_data_windows, input_target_data, \
                                 build_forecast_ts_training_dataset
from mlcf.datatools.wtseries import WTSeries


@pytest.fixture
def window_data(btc_ohlcv):
    win = WTSeries(window_width=10, raw_data=btc_ohlcv.iloc[0:100])
    return win


def test_split_pandas(btc_ohlcv):
    a, b = split_pandas(pd.DataFrame(columns=["a", "b"]))
    assert list(a.columns) == ["a", "b"] and list(b.columns) == ["a", "b"]
    data_a, data_b = split_pandas(btc_ohlcv.iloc[0:100])
    assert len(data_a) == len(data_b)
    assert data_a.index[49] == 49 and data_b.index[0] == 50

    data_a, data_b = split_pandas(btc_ohlcv.iloc[0:100], prop_snd_elem=0.0)
    assert len(data_b) == 0 and len(data_a) == 100

    data_a, data_b = split_pandas(btc_ohlcv.iloc[0:100], prop_snd_elem=1.0)
    assert len(data_b) == 100 and len(data_a) == 0

    with pytest.raises(Exception):
        split_pandas(btc_ohlcv.iloc[0:100], prop_snd_elem=1.1)

    with pytest.raises(Exception):
        split_pandas(btc_ohlcv.iloc[0:100], prop_snd_elem=-0.1)


def test_to_train_val_test(btc_ohlcv):
    train, val, test = to_train_val_test(btc_ohlcv.iloc[0:1000], prop_tv=0.2, prop_v=0.2)
    assert len(train) == 800 and len(val) == 40 and len(test) == 160
    assert train.index[790] == 790 and test.index[0] == 800 and test.index[150] == 950
    assert val.index[0] == 960


def test_split_in_interval(btc_ohlcv):
    l_i = split_in_interval(pd.DataFrame(columns=["a", "b"]))
    assert list(l_i[0].columns) == ["a", "b"]

    list_data = split_in_interval(btc_ohlcv.iloc[0:100], 1)
    assert len(list_data) == 1 and len(list_data[0]) == 100

    list_data = split_in_interval(btc_ohlcv.iloc[0:100], 2)
    assert len(list_data) == 2 and len(list_data[0]) == 50

    list_data = split_in_interval(btc_ohlcv.iloc[0:101], 3)
    assert len(list_data) == 3
    assert len(list_data[0]) == 34 and len(list_data[1]) == 34 and len(list_data[2]) == 33

    assert list_data[0].index[-1] == btc_ohlcv.iloc[0:100].index[-67]
    assert list_data[2].index[-1] == btc_ohlcv.iloc[0:101].index[-1]
    n = 11
    list_data = split_in_interval(btc_ohlcv.iloc[0:100], n)
    sum = np.sum([len(list_data[i]) for i in range(n)])
    assert sum == 100

    assert isinstance(list_data[0], pd.DataFrame)


def test_input_target_data_window(window_data):
    input, target = input_target_data_windows(window_data, 9, 1)
    assert len(input) == len(target) and len(input) == 100 - 10 + 1
    assert len(input[0]) == 9 and len(target[0]) == 1
    assert target[0].index[0] == input[1].index[-1]

    with pytest.raises(Exception):
        input, target = input_target_data_windows(window_data, 9, 2)

    input, target = input_target_data_windows(window_data, 8, 2)
    assert len(input[0]) == 8 and len(target[0]) == 2
    assert target[0].index[0] == input[2].index[-2] and target[0].index[1] == input[2].index[-1]

    assert isinstance(input[0], pd.DataFrame) and isinstance(target[0], pd.DataFrame)


def test_input_target_data(btc_ohlcv):
    input, target = input_target_data(btc_ohlcv.iloc[0:100], 9, 1)
    assert len(input) == 9 and len(target) == 1

    input, target = input_target_data(btc_ohlcv.iloc[0:100], 90, 10)
    assert len(input) == 90 and len(target) == 10

    with pytest.raises(Exception):
        input, target = input_target_data(btc_ohlcv.iloc[0:100], 90, 11)

    assert isinstance(input, pd.DataFrame) and isinstance(target, pd.DataFrame)


def test_build_forecast_ts_training_dataset(btc_ohlcv):
    ti, tt, vi, vt, tei, tet = build_forecast_ts_training_dataset(
        btc_ohlcv[["close", "open"]].iloc[0:1000],
        input_width=9,
        prop_tv=0.2,
        prop_v=0.2)
    assert np.all(ti[0] == btc_ohlcv[["close", "open"]].iloc[0:9])
    assert isinstance(ti, WTSeries)
    assert ti.ndim() == 2 and ti.ndim() == vt.ndim()
    assert ti.size() == (len(tt), 9, 2)
    assert len(ti) == 800 - 10 + 1

    ti, tt, vi, vt, tei, tet = build_forecast_ts_training_dataset(
        btc_ohlcv[["close", "open"]].iloc[0:1000],
        input_width=9,
        prop_tv=0.2,
        prop_v=0.2,
        do_shuffle=True)
    assert ti[0].index[0] == ti[0].index[1]-1
    assert ti[0].index[0] != ti[1].index[0]-1
