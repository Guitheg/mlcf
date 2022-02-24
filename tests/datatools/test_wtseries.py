import pytest
from mlcf.datatools.wtseries import window_data, WTSeries


def test_window_data(btc_ohlcv):
    list_data = window_data(btc_ohlcv.iloc[0:100], 10, 1)
    assert len(list_data[0]) == 10
    assert len(list_data) == ((100-10) // 1) + 1
    assert list_data[0].index[1] == list_data[1].index[0]

    list_data = window_data(btc_ohlcv.iloc[0:100], 5, 4)
    assert len(list_data[0]) == 5
    assert len(list_data) == ((100-5) // 4) + 1
    assert list_data[0].index[4] == list_data[1].index[0]

    list_data = window_data(btc_ohlcv.iloc[0:2], 10, 1)
    assert len(list_data[0]) == 0


def test_WTSeries(btc_ohlcv):
    win = WTSeries(10)
    assert win.is_empty()
    assert 0 == win.ndim()
    assert (0, 10, 0) == win.size()
    win2 = WTSeries(raw_data=btc_ohlcv[["close", "open"]].iloc[0:100], window_width=10)
    win.add_window_data(win2)
    assert win.ndim() == 2
    assert len(win) == len(win2)
    win3 = WTSeries(raw_data=btc_ohlcv.iloc[0:100], window_width=20)
    with pytest.raises(Exception):
        win.add_window_data(win3)
    win3.add_data(btc_ohlcv[100:200], window_step=5)
    assert len(win3) == (((100-20) // 1) + 1) + (((100-20) // 5) + 1)
    win3.add_one_window(btc_ohlcv[200:220])
    assert len(win3) == (((100-20) // 1) + 1) + (((100-20) // 5) + 1) + 1


def test_WTSeries_make_common_shuffle(btc_ohlcv):
    wts = WTSeries(raw_data=btc_ohlcv[["close", "open"]].iloc[0:100], window_width=20)
    wts2 = WTSeries(raw_data=btc_ohlcv[["close", "open"]].iloc[100:200], window_width=20)
    assert wts()[0].index[0] == 0
    assert wts2()[0].index[0] == 100
    wts.make_common_shuffle(wts2)
    assert wts()[0].index[0] != 0
    assert wts2()[0].index[0] != 100
    assert wts()[0].index[0] == wts2()[0].index[0]-100


def test_WTSeries_copy(btc_ohlcv):
    wts = WTSeries(raw_data=btc_ohlcv[["close", "open"]].iloc[0:100], window_width=20)
    wts_copy = wts.copy()
    assert wts._window_width == wts_copy._window_width
    assert wts.window_step == wts_copy.window_step
    assert len(wts.data) == len(wts_copy.data)
    for win, win_cp in zip(wts.data, wts_copy.data):
        assert len(win) == len(win_cp)
        assert len(win.columns) == len(win.columns)
