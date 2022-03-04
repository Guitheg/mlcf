from enum import Enum, unique
from functools import partial
from typing import List

# import pandas_ta as pta
import mlcf.datatools.indice_tools as i_tools
import numpy as np
import pandas as pd
import pandas_ta as _  # noqa
import talib.abstract as ta


@unique
class Indice(Enum):
    ADX = "ADX"

    # Plus Directional Indicator / Movement
    P_DIDM = "P_DIDM"

    # Minus Directional Indicator / Movement
    M_DIDM = "M_DIDM"

    # Aroon, Aroon Oscillator
    AROON = "AROON"

    # Awesome Oscillator
    AO = "AO"

    # Keltner Channel
    KELTNER = "KC"

    # Ultimate Oscillator
    UO = "UO"

    # Commodity Channel Index: values [Oversold:-100, Overbought:100]
    CCI = "CCI"
    RSI = "RSI"

    # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
    FISHER_RSI = "FISHER_RSI"

    # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
    FISHER_RSI_NORM = "FISHER_RSI_NORM"
    STOCH_SLOW = "STOCH_SLOW"
    STOCH_FAST = "STOCH_FAST"

    # Stochastic RSI
    # Please read https://github.com/freqtrade/freqtrade/issues/2961 before using this.
    # STOCHRSI is NOT aligned with tradingview, which may result in non-expected results.
    STOCH_RSI = "STOCH_RSI"

    MACD = "MACD"
    MFI = "MFI"
    ROC = "ROC"

    # Bollinger Bands
    BBANDS = "BBANDS"

    # Bollinger Bands - Weighted (EMA based instead of SMA)
    W_BBANDS = "W_BBANDS"

    # EMA - Exponential Moving Average
    EMA = "EMA"

    # SMA - Simple Moving Average
    SMA = "SMA"

    # Parabolic SAR
    SAR = "SAR"

    # TEMA - Triple Exponential Moving Average
    TEMA = "TEMA"

    # Hilbert Transform Indicator - SineWave
    HT = "HT"

    # -- Pattern Recognition - Bullish candlestick patterns --
    # Hammer: values [0, 100]
    HAMM = "HAMM"

    # Inverted Hammer: values [0, 100]
    IHAMM = "IHAMM"

    # Dragonfly Doji: values [0, 100]
    DRAGDOJI = "DRAGDOJI"

    # Piercing Line: values [0, 100]
    PIERCINGLINE = "PIERCINGLINE"

    # Morningstar: values [0, 100]
    MORNINGSTAR = "MORNINGSTAR"

    # Three White Soldiers: values [0, 100]
    TWSOLDIER = "TWSOLDIER"

    # -- Pattern Recognition - Bearish candlestick patterns --
    # Hanging Man: values [0, 100]
    HANGMAN = "HANGMAN"

    # Shooting Star: values [0, 100]
    SHOOTSTAR = "SHOOTSTAR"

    # Gravestone Doji: values [0, 100]
    GRAVESTONE = "GRAVESTONE"

    # Dark Cloud Cover: values [0, 100]
    DARKCLOUD = "DARKCLOUD"

    # Evening Doji Star: values [0, 100]
    EVENDOJISTAR = "EVENDOJISTAR"

    # Evening Star: values [0, 100]
    EVENSTAR = "EVENSTAR"

    # -- Pattern Recognition - Bullish/Bearish candlestick patterns --
    # Three Line Strike: values [0, -100, 100]
    TLSTRIKE = "TLSTRIKE"

    # Spinning Top: values [0, -100, 100]
    SPINTOP = "SPINTOP"

    # Engulfing: values [0, -100, 100]
    ENGULFING = "ENGULFING"

    # Harami: values [0, -100, 100]
    HARAMI = "HARAMI"

    # Three Outside Up/Down: values [0, -100, 100]
    THREEOUTUPDOWN = "THREEOUTUPDOWN"

    # Three Inside Up/Down: values [0, -100, 100]
    THREEINUPDOWN = "THREEINUPDOWN"
    ###

    # Heikin Ashi Strategy
    HEIKINASHI = "HEIKINASHI"

    # Ichimoku Kinkō Hyō (ichimoku)
    ICHIMOKU = "ICHIMOKU"

    # /////////____________CUSTOM INDICATORS STARTS HERE____________/////////

    # percent growth
    PERCENTGROWTH = "PERCENTGROWTH"

    # SMA1
    SMA1 = "SMA1"

    # log of SMA1
    LNSMA1 = "LNSMA1"

    # volatility
    VOLATILITY = "VOLATILITY"

    # return
    RETURN = "RETURN"

    @classmethod
    def list_value(self):
        return [item.value for item in list(self)]


def switch_indice(match: Indice, value: Indice):
    return match.value == value.value


def add_indicator(data: pd.DataFrame, indice_name: Indice):
    dataframe = data.copy()

    case = partial(switch_indice, indice_name)

    if case(Indice.ADX):
        dataframe["adx"] = ta.ADX(dataframe)

    elif case(Indice.P_DIDM):
        dataframe["plus_dm"] = ta.PLUS_DM(dataframe)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)

    elif case(Indice.M_DIDM):
        dataframe["minus_dm"] = ta.MINUS_DM(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)

    elif case(Indice.AROON):
        aroon = ta.AROON(dataframe)
        dataframe["aroonup"] = aroon["aroonup"]
        dataframe["aroondown"] = aroon["aroondown"]
        dataframe["aroonosc"] = ta.AROONOSC(dataframe)

    elif case(Indice.AO):
        dataframe["ao"] = i_tools.awesome_oscillator(dataframe)

    elif case(Indice.KELTNER):
        keltner = i_tools.keltner_channel(dataframe)
        dataframe["kc_upperband"] = keltner["upper"]
        dataframe["kc_lowerband"] = keltner["lower"]
        dataframe["kc_middleband"] = keltner["mid"]
        dataframe["kc_percent"] = (dataframe["close"] - dataframe["kc_lowerband"]) / (
            dataframe["kc_upperband"] - dataframe["kc_lowerband"]
        )
        dataframe["kc_width"] = (
            dataframe["kc_upperband"] - dataframe["kc_lowerband"]
        ) / dataframe["kc_middleband"]

    elif case(Indice.UO):
        dataframe["uo"] = ta.ULTOSC(dataframe)

    elif case(Indice.CCI):
        dataframe["cci"] = ta.CCI(dataframe)

    elif case(Indice.RSI):
        dataframe["rsi"] = ta.RSI(dataframe)

    elif case(Indice.FISHER_RSI):
        rsi = 0.1 * (dataframe["rsi"] - 50)
        dataframe["fisher_rsi"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

    elif case(Indice.FISHER_RSI_NORM):
        dataframe["fisher_rsi_norma"] = 50 * (dataframe["fisher_rsi"] + 1)

    elif case(Indice.STOCH_SLOW):
        stoch = ta.STOCH(dataframe)
        dataframe["slowd"] = stoch["slowd"]
        dataframe["slowk"] = stoch["slowk"]

    elif case(Indice.STOCH_FAST):
        stoch_fast = ta.STOCHF(dataframe)
        dataframe["fastd"] = stoch_fast["fastd"]
        dataframe["fastk"] = stoch_fast["fastk"]

    elif case(Indice.STOCH_RSI):
        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe["fastd_rsi"] = stoch_rsi["fastd"]
        dataframe["fastk_rsi"] = stoch_rsi["fastk"]

    elif case(Indice.MACD):
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

    elif case(Indice.MFI):
        dataframe["mfi"] = ta.MFI(dataframe)

    elif case(Indice.ROC):
        dataframe["roc"] = ta.ROC(dataframe)

    elif case(Indice.BBANDS):
        bollinger = i_tools.bollinger_bands(
            i_tools.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lowerband"]) / (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        )
        dataframe["bb_width"] = (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        ) / dataframe["bb_middleband"]

    elif case(Indice.W_BBANDS):
        weighted_bollinger = i_tools.weighted_bollinger_bands(
            i_tools.typical_price(dataframe), window=20, stds=2
        )
        dataframe["wbb_upperband"] = weighted_bollinger["upper"]
        dataframe["wbb_lowerband"] = weighted_bollinger["lower"]
        dataframe["wbb_middleband"] = weighted_bollinger["mid"]
        dataframe["wbb_percent"] = (dataframe["close"] - dataframe["wbb_lowerband"]) / (
            dataframe["wbb_upperband"] - dataframe["wbb_lowerband"]
        )
        dataframe["wbb_width"] = (
            dataframe["wbb_upperband"] - dataframe["wbb_lowerband"]
        ) / dataframe["wbb_middleband"]

    elif case(Indice.EMA):
        dataframe["ema3"] = ta.EMA(dataframe, timeperiod=3)
        dataframe["ema5"] = ta.EMA(dataframe, timeperiod=5)
        dataframe["ema10"] = ta.EMA(dataframe, timeperiod=10)
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema100"] = ta.EMA(dataframe, timeperiod=100)

    elif case(Indice.SMA):
        dataframe["sma3"] = ta.SMA(dataframe, timeperiod=3)
        dataframe["sma5"] = ta.SMA(dataframe, timeperiod=5)
        dataframe["sma10"] = ta.SMA(dataframe, timeperiod=10)
        dataframe["sma21"] = ta.SMA(dataframe, timeperiod=21)
        dataframe["sma50"] = ta.SMA(dataframe, timeperiod=50)
        dataframe["sma100"] = ta.SMA(dataframe, timeperiod=100)

    elif case(Indice.SAR):
        dataframe["sar"] = ta.SAR(dataframe)

    elif case(Indice.TEMA):
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

    elif case(Indice.HT):
        hilbert = ta.HT_SINE(dataframe)
        dataframe["htsine"] = hilbert["sine"]
        dataframe["htleadsine"] = hilbert["leadsine"]

    elif case(Indice.HAMM):
        dataframe["CDLHAMMER"] = ta.CDLHAMMER(dataframe)

    elif case(Indice.IHAMM):
        dataframe["CDLINVERTEDHAMMER"] = ta.CDLINVERTEDHAMMER(dataframe)

    elif case(Indice.DRAGDOJI):
        dataframe["CDLDRAGONFLYDOJI"] = ta.CDLDRAGONFLYDOJI(dataframe)

    elif case(Indice.PIERCINGLINE):
        dataframe["CDLPIERCING"] = ta.CDLPIERCING(dataframe)

    elif case(Indice.MORNINGSTAR):
        dataframe["CDLMORNINGSTAR"] = ta.CDLMORNINGSTAR(dataframe)

    elif case(Indice.TWSOLDIER):
        dataframe["CDL3WHITESOLDIERS"] = ta.CDL3WHITESOLDIERS(dataframe)

    elif case(Indice.HANGMAN):
        dataframe["CDLHANGINGMAN"] = ta.CDLHANGINGMAN(dataframe)

    elif case(Indice.SHOOTSTAR):
        dataframe["CDLSHOOTINGSTAR"] = ta.CDLSHOOTINGSTAR(dataframe)

    elif case(Indice.GRAVESTONE):
        dataframe["CDLGRAVESTONEDOJI"] = ta.CDLGRAVESTONEDOJI(dataframe)

    elif case(Indice.DARKCLOUD):
        dataframe["CDLDARKCLOUDCOVER"] = ta.CDLDARKCLOUDCOVER(dataframe)

    elif case(Indice.EVENDOJISTAR):
        dataframe["CDLEVENINGDOJISTAR"] = ta.CDLEVENINGDOJISTAR(dataframe)

    elif case(Indice.EVENSTAR):
        dataframe["CDLEVENINGSTAR"] = ta.CDLEVENINGSTAR(dataframe)

    elif case(Indice.TLSTRIKE):
        dataframe["CDL3LINESTRIKE"] = ta.CDL3LINESTRIKE(dataframe)

    elif case(Indice.SPINTOP):
        dataframe["CDLSPINNINGTOP"] = ta.CDLSPINNINGTOP(dataframe)

    elif case(Indice.ENGULFING):
        dataframe["CDLENGULFING"] = ta.CDLENGULFING(dataframe)

    elif case(Indice.HARAMI):
        dataframe["CDLHARAMI"] = ta.CDLHARAMI(dataframe)

    elif case(Indice.THREEOUTUPDOWN):
        dataframe["CDL3OUTSIDE"] = ta.CDL3OUTSIDE(dataframe)

    elif case(Indice.THREEINUPDOWN):
        dataframe["CDL3INSIDE"] = ta.CDL3INSIDE(dataframe)  # values [0, -100, 100]

    elif case(Indice.HEIKINASHI):
        heikinashi = i_tools.heikinashi(dataframe)
        dataframe["ha_open"] = heikinashi["open"]
        dataframe["ha_close"] = heikinashi["close"]
        dataframe["ha_high"] = heikinashi["high"]
        dataframe["ha_low"] = heikinashi["low"]

    elif case(Indice.ICHIMOKU):
        ichimoku, ichimoku_forward = dataframe.ta.ichimoku(lookahead=False)
        dataframe["ich_tenkan"] = ichimoku["ITS_9"]
        dataframe["ich_kijun"] = ichimoku["IKS_26"]
        dataframe["ich_spanA"] = ichimoku["ISA_9"]
        dataframe["ich_spanB"] = ichimoku["ISB_26"]
        dataframe["ich_chikou"] = dataframe["close"].shift(26)

        # Warning: The use of shift(-26) is only to simulate the leading span A and B because they
        # are in the future (so it's normal).
        # 26 candles forward
        # I duplicate spanA and spanB, shift it by -26 and add at the end leading span a and
        # leading span b.

        dataframe["ich_lead_spanA"] = pd.concat(
            [ichimoku["ISA_9"], ichimoku_forward["ISA_9"]]
        ).shift(-26)
        dataframe["ich_lead_spanB"] = pd.concat(
            [ichimoku["ISB_26"], ichimoku_forward["ISB_26"]]
        ).shift(-26)

    elif case(Indice.SMA1):
        dataframe = add_SMA1(dataframe)

    elif case(Indice.VOLATILITY):
        dataframe = add_volatility(dataframe)

    elif case(Indice.LNSMA1):
        dataframe = add_ln_SMA1(dataframe)

    elif case(Indice.PERCENTGROWTH):
        dataframe = add_percent_growth(dataframe)

    elif case(Indice.RETURN):
        dataframe = add_return(dataframe)

    else:
        raise Exception("Unknown indice")

    return dataframe


def add_indicators(data: pd.DataFrame, list_indice: List[Indice], dropna: bool = True):
    dataframe = data.copy()

    for indice in list_indice:
        dataframe = add_indicator(dataframe, indice)

    dataframe.dropna(inplace=dropna)
    return dataframe


def add_percent_growth(data: pd.DataFrame):  # add percent growth on close

    drop_SMA1 = False
    if 'SMA1' not in data:
        dataframe = add_SMA1(data.copy())
        drop_SMA1 = True
    else:
        dataframe = data.copy()

    growth_offset_list = [1, 3, 5]

    for offset in growth_offset_list:
        SMA1_copy = dataframe['SMA1'].copy().shift(offset)
        dataframe['growth'+str(offset)] = dataframe['SMA1'].div(SMA1_copy)

    if drop_SMA1:
        dataframe.drop(['SMA1'], axis=1, inplace=True)
    return dataframe


def add_SMA1(data: pd.DataFrame):  # add percent growth on close
    dataframe = data.copy()

    dataframe['SMA1'] = (
        dataframe['close'] +
        dataframe['high'] +
        dataframe['low'] +
        dataframe['open']
    )/4

    return dataframe


def add_ln_SMA1(data: pd.DataFrame):
    drop_SMA1 = False
    if 'SMA1' not in data:
        dataframe = add_SMA1(data.copy())
        drop_SMA1 = True
    else:
        dataframe = data.copy()

    dataframe['lnSMA1'] = np.log(dataframe['SMA1'])

    if drop_SMA1:
        dataframe.drop('SMA1', axis=1, inplace=True)
    return dataframe


def add_volatility(data: pd.DataFrame):

    volatility_offset_list = [1, 3, 5]
    drop_growth = False
    if ('growth'+str(volatility_offset_list[0])) not in data:
        dataframe = add_percent_growth(data.copy())
        drop_growth = True
    else:
        dataframe = data.copy()

    for offset in volatility_offset_list:
        dataframe['volatility'+str(offset)] = np.log(dataframe['growth'+str(offset)])

    if drop_growth:
        for offset in volatility_offset_list:
            if ('growth'+str(offset)) in data:
                dataframe.drop(('growth'+str(offset)), axis=1, inplace=True)
    return dataframe


def add_return(data: pd.DataFrame, offset: int = 1, colname: str = "return", dropna: bool = False):
    dataframe = data.copy()
    dataframe[colname] = np.log(dataframe.close) - np.log(dataframe.shift(offset).close)
    dataframe.dropna(inplace=dropna)
    return dataframe
