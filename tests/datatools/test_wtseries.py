import pytest
from datatools.wtseries import window_data, WTSeries
import pandas as pd

def init_data():
    data = pd.read_json("tests/testdata/BTC_BUSD-1h.json")
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data.values, columns=columns)
    data['date'] =  pd.to_datetime(data["date"], unit="ms")
    return data
data = init_data()

def test_window_data():
    list_data = window_data(data.iloc[0:100], 10, 1)
    assert len(list_data[0]) == 10
    assert len(list_data) == ((100-10) // 1) + 1
    assert list_data[0].index[1] == list_data[1].index[0]
    
    list_data = window_data(data.iloc[0:100], 5, 4)
    assert len(list_data[0]) == 5
    assert len(list_data) == ((100-5) // 4) + 1
    assert list_data[0].index[4] == list_data[1].index[0]
    
    with pytest.raises(Warning):
        list_data = window_data(data.iloc[0:2], 10, 1)
        assert len(list_data[0]) == 0
        
def test_WTSeries():
    win = WTSeries(10)
    assert True == win.is_empty()
    assert 0 == win.ndim()
    assert (0, 10, 0) == win.size()
    win2 = WTSeries(data=data[["close", "open"]].iloc[0:100], window_width=10)
    win.merge_window_data(win2)
    assert win.ndim() == 2
    assert len(win) == len(win2)
    win3 = WTSeries(data=data.iloc[0:100], window_width=20)
    with pytest.raises(Exception):
        win.merge_window_data(win3)
    win3.add_data(data[100:200], 5)
    assert len(win3) == (((100-20) // 1) + 1) + (((100-20) // 5) + 1)
    win3.add_one_window(data[200:220])
    assert len(win3) == (((100-20) // 1) + 1) + (((100-20) // 5) + 1) + 1
    