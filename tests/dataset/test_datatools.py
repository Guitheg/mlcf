import pytest
import pandas as pd
from dataset.datatools import split_pandas, to_train_val_test


def init_data():
    data = pd.read_json("tests/testdata/BTC_BUSD-1d.json")
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data.values, columns=columns)
    data['date'] =  pd.to_datetime(data["date"], unit="ms")
    return data

data = init_data()

def test_split_pandas():
    data_a, data_b = split_pandas(data.iloc[0:100])
    assert len(data_a) == len(data_b)
    assert data_a.index[49] == 49 and data_b.index[0] == 50
    # 
    data_a, data_b = split_pandas(data.iloc[0:101])
    assert len(data_a) == len(data_b)+1
    
    data_a, data_b = split_pandas(data.iloc[0:100], prop_snd_elem=0.9)
    assert len(data_b) == 90 and len(data_a) == 10

    data_a, data_b = split_pandas(data.iloc[0:100], prop_snd_elem=0.1)
    assert len(data_b) == 10 and len(data_a) == 90
    
    data_a, data_b = split_pandas(data.iloc[0:100], prop_snd_elem=0.0)
    assert len(data_b) == 0 and len(data_a) == 100
    
    data_a, data_b = split_pandas(data.iloc[0:100], prop_snd_elem=1.0)
    assert len(data_b) == 100 and len(data_a) == 0

    with pytest.raises(Exception):
        split_pandas(data.iloc[0:100], prop_snd_elem=1.1)
    
    with pytest.raises(Exception):
        split_pandas(data.iloc[0:100], prop_snd_elem=-0.1)


def test_to_train_val_test():
    train, val, test = to_train_val_test(data.iloc[0:100], test_val_prop = 0.2, val_prop = 0.2)
    assert len(train) == 80 and len(val) == 4 and len(test) == 16
    assert train.index[79] == 79 and test.index[0] == 80 and test.index[15] == 95 
    assert val.index[0] == 96