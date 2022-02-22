
from mlcf.aitools.super_module import SuperModule
from mlcf.datatools.wtst import WTSTraining
import pandas as pd


def init_data():
    data = pd.read_json("tests/testdata/BTC_BUSD-1h.json")
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data.values, columns=columns)
    data['date'] = pd.to_datetime(data["date"], unit="ms")
    return data


data = init_data()


def test_super_module():
    ts_data = WTSTraining(20, index_column="date")
    SuperModule(ts_data.input_width)
