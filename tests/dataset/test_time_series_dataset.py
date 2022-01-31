import pytest
import pandas as pd
from dataset.time_series_dataset import Time_Series_Dataset

def init_data():
    data = pd.read_json("tests/testdata/BTC_BUSD-1h.json")
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data.values, columns=columns)
    data['date'] =  pd.to_datetime(data["date"], unit="ms")
    return data
data = init_data()


def test_Time_Series_Dataset():
    ts_data = Time_Series_Dataset(9)
    ts_data.add_time_serie(data.iloc[0:1000], test_val_prop=0.2)
    assert len(ts_data.x_train()) == 800 - 10 + 1