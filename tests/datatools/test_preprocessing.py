from ctbt.datatools.preprocessing import Identity, AutoNormalize
from ctbt.datatools.wtseries import WTSeries
import numpy as np
import pandas as pd

def init_data():
    data = np.arange(1000)
    columns = ["value"]
    data = pd.DataFrame(data, columns=columns)
    return data
data = init_data()

def test_Identity():
    wtseries = WTSeries(10, data.iloc[0:1000])
    preprocess = Identity(wtseries)
    prep_data = preprocess()
    for i in range(len(wtseries)):
        assert np.all(prep_data[i] == wtseries[i])
    
def test_AutoNormalize():
    wtseries = WTSeries(10, data.iloc[0:1000])
    preprocess = AutoNormalize(wtseries)
    prep_data = preprocess()
    for i in range(len(wtseries)):
        y = ((wtseries[i] - wtseries[i].mean()) / wtseries[i].std()).round(6)
        assert np.all(prep_data[i] == y)
        assert isinstance(prep_data[i], pd.DataFrame)