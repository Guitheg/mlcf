from ai.super_module import SuperModule
import ai
from datatools.wtseries_training import WTSeriesTraining
from datatools import utils
import pandas as pd
import pytest
import os

def init_data():
    data = pd.read_json("tests/testdata/BTC_BUSD-1h.json")
    columns = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data.values, columns=columns)
    data['date'] =  pd.to_datetime(data["date"], unit="ms")
    return data
data = init_data()

def test_super_module(mocker):
    ts_data = WTSeriesTraining(20, column_index="date")
    module = SuperModule(ts_data.input_size)
     