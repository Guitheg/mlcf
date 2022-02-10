
from os.path import isdir, isfile, join
import pickle   
import pandas as pd
from CGrbi.datatools.wtseries_training import WTSeriesTraining, EXTENSION_FILE, read_wtseries_training

def init_data():
    data = pd.read_json("tests/testdata/BTC_BUSD-1h.json")
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data.values, columns=columns)
    data['date'] =  pd.to_datetime(data["date"], unit="ms")
    return data
data = init_data()

def test_WTSeriesTraining():
    ts_data = WTSeriesTraining(9)
    ts_data.add_time_serie(data.iloc[0:1000], test_val_prop=0.2)
    assert len(ts_data.x_train()) == 800 - 10 + 1
    
def test_io_WTSeriesTraining(mocker):
    ts_data = WTSeriesTraining(9, index_column='date')
    ts_data.add_time_serie(data.iloc[0:1000], test_val_prop=0.2)
    ts_data.write("tests/testdata", "WTStraining_BTCBUSD-1h")
    ts_data_2 = read_wtseries_training(
        join("tests/testdata", "WTStraining_BTCBUSD-1h"+EXTENSION_FILE))
    assert ts_data.input_size == ts_data_2.input_size
    assert ts_data.index_column == ts_data_2.index_column