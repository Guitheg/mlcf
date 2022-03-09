from mlcf.datatools.indice import add_indicators, add_indicator, Indice
import numpy as np


def test_add_indicator(btc_ohlcv):
    data = add_indicator(btc_ohlcv, Indice.ADX)
    assert len(data.columns) == 7


def test_add_indicators(btc_ohlcv):
    assert len(btc_ohlcv.columns) == 6
    list_indice = [Indice.ADX, Indice.P_DIM]
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


def test_stats(btc15m_ohlcv):
    data = add_indicator(btc15m_ohlcv, Indice.STATS)
    data.dropna(inplace=True)
    assert len(data) == len(data[~data.isin([np.inf, -np.inf]).any(1)])
    assert len(data) != 0


def test_indicators_list_to_std(btc_ohlcv):
    list_indice = list(Indice)
    list_to_std = set()
    dataframe = add_indicators(btc_ohlcv, list_indice, standardize=True, list_to_std=list_to_std)
    assert len(list_to_std) == 127
    assert (dataframe["adx"] >= 0).all() and (dataframe["adx"] <= 1).all()
    assert (dataframe["plus_di"] >= 0).all() and (dataframe["plus_di"] <= 1).all()
    assert (dataframe["minus_di"] >= 0).all() and (dataframe["minus_di"] <= 1).all()
    assert (dataframe["aroonup"] >= 0).all() and (dataframe["aroonup"] <= 1).all()
    assert (dataframe["aroondown"] >= 0).all() and (dataframe["aroondown"] <= 1).all()
    assert (dataframe["aroonosc"] >= -1).all() and (dataframe["aroonosc"] <= 1).all()
    assert (dataframe["uo"] >= 0).all() and (dataframe["uo"] <= 1).all()
    assert (dataframe["rsi"] >= 0).all() and (dataframe["rsi"] <= 1).all()
    assert (dataframe["fisher_rsi"] >= -1).all() and (dataframe["fisher_rsi"] <= 1).all()
    assert (dataframe["slowd"] >= 0).all() and (dataframe["slowd"] <= 1).all()
    assert (dataframe["slowk"] >= 0).all() and (dataframe["slowk"] <= 1).all()
    assert (dataframe["fastd"] >= 0).all() and (dataframe["fastd"] <= 1).all()
    assert (dataframe["fastk"] >= 0).all() and (dataframe["fastk"] <= 1).all()
    assert (dataframe["fastd_rsi"] >= 0).all() and (dataframe["fastd_rsi"] <= 1).all()
    assert (dataframe["fastk_rsi"] >= 0).all() and (dataframe["fastk_rsi"] <= 1).all()
    assert (dataframe["mfi"] >= 0).all() and (dataframe["mfi"] <= 1).all()
    assert (dataframe["htsine"] >= -1).all() and (dataframe["htsine"] <= 1).all()
    assert (dataframe["htleadsine"] >= -1).all() and (dataframe["htleadsine"] <= 1).all()
