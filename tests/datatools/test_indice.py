from ctbt.datatools.indice import add_indicators, add_indicator, Indice
import pandas as pd

def init_data():
    data = pd.read_json("tests/testdata/BTC_BUSD-1h.json")
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data.values, columns=columns)
    data['date'] =  pd.to_datetime(data["date"], unit="ms")
    return data


def test_add_indicator():
    data = init_data()
    data = add_indicator(data, Indice.ADX)
    assert len(data.columns) == 7
    
def test_add_indicators():
    data = init_data()
    assert len(data.columns) == 6
    list_indice = [Indice.ADX, Indice.P_DIDM]
    data = add_indicators(data, list_indice)
    assert len(data.columns) == 9