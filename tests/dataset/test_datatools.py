import pytest
import pandas as pd
import numpy as np
from dataset.datatools import split_pandas, to_train_val_test, \
    split_in_interval, input_target_data_windows, input_target_data, \
    make_commmon_shuffle, build_forecast_ts_training_dataset
    
from dataset.window_data import Window_Data, window_data


def init_data():
    data = pd.read_json("tests/testdata/BTC_BUSD-1h.json")
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data.values, columns=columns)
    data['date'] =  pd.to_datetime(data["date"], unit="ms")
    return data.iloc[0:2000]
data = init_data()

def init_window_data():
    win = window_data(data.iloc[0:100], 10)
    return win
win_data = init_window_data()

def test_split_pandas():
    data_a, data_b = split_pandas(data.iloc[0:100])
    assert len(data_a) == len(data_b)
    assert data_a.index[49] == 49 and data_b.index[0] == 50
    
    data_a, data_b = split_pandas(data.iloc[0:100], prop_snd_elem=0.0)
    assert len(data_b) == 0 and len(data_a) == 100
    
    data_a, data_b = split_pandas(data.iloc[0:100], prop_snd_elem=1.0)
    assert len(data_b) == 100 and len(data_a) == 0

    with pytest.raises(Exception):
        split_pandas(data.iloc[0:100], prop_snd_elem=1.1)
    
    with pytest.raises(Exception):
        split_pandas(data.iloc[0:100], prop_snd_elem=-0.1)


def test_to_train_val_test():
    train, val, test = to_train_val_test(data.iloc[0:1000], test_val_prop = 0.2, val_prop = 0.2)
    assert len(train) == 800 and len(val) == 40 and len(test) == 160
    assert train.index[790] == 790 and test.index[0] == 800 and test.index[150] == 950 
    assert val.index[0] == 960
    
def test_split_in_interval():
    list_data = split_in_interval(data.iloc[0:100], 1)
    assert len(list_data) == 1 and len(list_data[0]) == 100
    
    list_data = split_in_interval(data.iloc[0:100], 2)
    assert len(list_data) == 2 and len(list_data[0]) == 50
    
    list_data = split_in_interval(data.iloc[0:101], 3)
    assert len(list_data) == 3
    assert len(list_data[0]) == 34 and len(list_data[1]) == 34 and len(list_data[2]) == 33
    
    assert list_data[0].index[-1] == data.iloc[0:100].index[-67]
    assert list_data[2].index[-1] == data.iloc[0:101].index[-1]
    n = 11
    list_data = split_in_interval(data.iloc[0:100], n)
    sum = np.sum([len(list_data[i]) for i in range(n)])
    assert sum == 100
    
    assert isinstance(list_data[0], pd.DataFrame)
    
def test_input_target_data_window():
    input, target = input_target_data_windows(win_data, 9, 1)
    assert len(input) == len(target) and len(input) == 100 - 10 + 1
    assert len(input[0]) == 9 and len(target[0]) == 1
    assert target[0].index[0] == input[1].index[-1]
    
    with pytest.raises(Exception):
        input, target = input_target_data_windows(win_data, 9, 2)
        
    input, target = input_target_data_windows(win_data, 8, 2)
    assert len(input[0]) == 8 and len(target[0]) == 2
    assert target[0].index[0] == input[2].index[-2] and target[0].index[1] == input[2].index[-1]
    
    assert isinstance(input[0], pd.DataFrame) and isinstance(target[0], pd.DataFrame)

def test_input_target_data():
    input, target = input_target_data(data.iloc[0:100], 9, 1)
    assert len(input) == 9 and len(target) == 1
    
    input, target = input_target_data(data.iloc[0:100], 90, 10)
    assert len(input) == 90 and len(target) == 10
    
    with pytest.raises(Exception):
        input, target = input_target_data(data.iloc[0:100], 90, 11)
        
    assert isinstance(input, pd.DataFrame) and isinstance(target, pd.DataFrame)

def test_make_common_shuffle():
    i = [1,2,3,4,5,6,7]
    c = [7,6,5,4,3,2,1]
    a, b = make_commmon_shuffle(i,c)
    for i in range(7):
        assert a[i] == 8-b[i]
        
def test_build_forecast_ts_training_dataset():
    ti, tt, vi, vt, tei, tet = build_forecast_ts_training_dataset(
        data[["close","open"]].iloc[0:1000], 
        input_width = 9, 
        test_val_prop=0.2, 
        val_prop=0.2)
    assert isinstance(ti, Window_Data)
    assert ti.n_features() == 2 and ti.n_features() == vt.n_features()
    assert ti.shape() == (len(tt), 9, 2)
    assert len(ti) == 800 - 10 + 1