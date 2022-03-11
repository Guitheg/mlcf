from mlcf.datatools.preprocessing import Identity, AutoNormalize
from mlcf.datatools.wtseries import WTSeries
import numpy as np
import pandas as pd


def test_Identity(btc_ohlcv):
    btc_ohlcv.set_index("date", inplace=True)
    wtseries = WTSeries(10, btc_ohlcv.iloc[0:1000])
    preprocess = Identity(wtseries)
    prep_data = preprocess()
    for i in range(len(wtseries)):
        assert np.all(prep_data[i] == wtseries[i])


def test_AutoNormalize(btc15m_ohlcv):
    btc15m_ohlcv.set_index("date", inplace=True)
    wtseries = WTSeries(10, btc15m_ohlcv.iloc[0:100])
    preprocess = AutoNormalize(wtseries)
    prep_data = preprocess()
    y = [
        ((w - w.mean()) / w.std()) for w in wtseries
        if not ((w - w.mean()) / w.std()).isnull().values.any()]
    assert len(y) == len(prep_data)
    for i, _ in enumerate(y):
        assert np.all(prep_data[i].round(3) == y[i].round(3))
        assert isinstance(prep_data[i], pd.DataFrame)
