from mlcf.datatools.indice import add_indicators, add_indicator, Indice
import numpy as np

def test_add_indicator(btc_ohlcv):
    data = add_indicator(btc_ohlcv, Indice.ADX)
    assert len(data.columns) == 7


def test_add_indicators(btc_ohlcv):
    assert len(btc_ohlcv.columns) == 6
    list_indice = [Indice.ADX, Indice.P_DIDM]
    data = add_indicators(btc_ohlcv, list_indice)
    assert len(data.columns) == 9


def test_add_all_indicators(btc_ohlcv):
    list_indice = list(Indice)
    _ = add_indicators(btc_ohlcv, list_indice)


def test_add_bbands(btc15m_ohlcv):
    data = add_indicator(btc15m_ohlcv, Indice.BBANDS)
    data.dropna(inplace=True)
    assert len(data) == len(data[~data.isin([np.inf, -np.inf]).any(1)])


def test_keltner_channel(btc15m_ohlcv):
    data = add_indicator(btc15m_ohlcv, Indice.KELTNER)
    data.dropna(inplace=True)
    assert len(data) == len(data[~data.isin([np.inf, -np.inf]).any(1)])


def test_add_wbbands(btc15m_ohlcv):
    data = add_indicator(btc15m_ohlcv, Indice.W_BBANDS)
    data.dropna(inplace=True)
    assert len(data) == len(data[~data.isin([np.inf, -np.inf]).any(1)])
