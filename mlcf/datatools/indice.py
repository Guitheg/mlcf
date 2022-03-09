from mlcf.utils import ListEnum
from enum import unique
from typing import Callable, Dict, List, Set, Tuple

from sklearn.preprocessing import MinMaxScaler

# import pandas_ta as pta
import mlcf.datatools.indice_tools as i_tools
import numpy as np
import pandas as pd
import pandas_ta as _  # noqa
import talib.abstract as ta


def min_max_scale(series, minmax: Tuple[float, float], feature_range: Tuple[float, float] = (0, 1)):
    mmsc = MinMaxScaler(feature_range=feature_range)
    mmsc.fit([[minmax[0]], [minmax[1]]])
    data = series.copy()
    data.loc[:] = mmsc.transform(data.values.reshape(-1, 1)).reshape(-1)
    return data


@unique
class Indice(ListEnum):
    ADX = "ADX"

    # Plus Directional Indicator / Movement
    P_DIM = "P_DIM"

    # Minus Directional Indicator / Movement
    M_DIM = "M_DIM"

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

    # Pattern Recognition indicators
    PATTERNS = "PATTERNS"

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

    # return
    RETURN = "RETURN"

    CANDLE_DIR = "CANDLE_DIR"

    CANDLE_HEIGHT = "CANDLE_HEIGHT"

    STATS = "STATS"


def switch_indice(match: Indice, value: Indice):
    return match.value == value.value


def add_indicator(
    data: pd.DataFrame,
    indice_name: Indice,
    standardize: bool = True,
    list_to_std: Set[str] = set()
):
    dataframe = data.copy()

    IndiceDict: Dict[str, Callable] = {
        Indice.ADX.value: add_adx,
        Indice.P_DIM.value: add_p_dim,
        Indice.M_DIM.value: add_m_dim,
        Indice.AROON.value: add_aroon,
        Indice.AO.value: add_ao,
        Indice.KELTNER.value: add_keltner,
        Indice.UO.value: add_uo,
        Indice.CCI.value: add_cci,
        Indice.RSI.value: add_rsi,
        Indice.FISHER_RSI.value: add_fisher_rsi,
        Indice.STOCH_SLOW.value: add_stochastic_slow,
        Indice.STOCH_FAST.value: add_stochastic_fast,
        Indice.STOCH_RSI.value: add_stoch_rsi,
        Indice.MACD.value: add_macd,
        Indice.MFI.value: add_mfi,
        Indice.ROC.value: add_roc,
        Indice.BBANDS.value: add_bbands,
        Indice.W_BBANDS.value: add_wbbands,
        Indice.EMA.value: add_ema,
        Indice.SMA.value: add_sma,
        Indice.SAR.value: add_sar,
        Indice.TEMA.value: add_tema,
        Indice.HT.value: add_hilbert,
        Indice.HEIKINASHI.value: add_heikinashi,
        Indice.ICHIMOKU.value: add_ichimoku,
        Indice.PATTERNS.value: add_pattern_recognition_indicators,
        Indice.SMA1.value: add_SMA1,
        Indice.LNSMA1.value: add_ln_SMA1,
        Indice.PERCENTGROWTH.value: add_percent_growth,
        Indice.RETURN.value: add_return,
        Indice.CANDLE_DIR.value: add_candle_dir,
        Indice.CANDLE_HEIGHT.value: add_candle_heigth,
        Indice.STATS.value: add_stats_indicators_on_rolling_windows
    }

    dataframe = IndiceDict[indice_name.value](
        data=dataframe,
        standardize=standardize,
        list_to_std=list_to_std
    )

    return dataframe


def add_indicators(
    data: pd.DataFrame,
    list_indice: List[Indice],
    dropna: bool = True,
    standardize: bool = True,
    list_to_std: Set[str] = set()
):
    dataframe = data.copy()

    for indice in list_indice:
        dataframe = add_indicator(dataframe, indice, standardize, list_to_std)

    dataframe.dropna(inplace=dropna)
    return dataframe


def add_adx(data: pd.DataFrame, standardize: bool = False, *args, **kwargs):
    dataframe = data.copy()
    adx = ta.ADX(data)

    if standardize:
        adx = min_max_scale(adx, (0, 100)).round(8)

    dataframe["adx"] = adx
    return dataframe


def add_p_dim(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
    plus_dm = ta.PLUS_DM(dataframe)
    plus_di = ta.PLUS_DI(dataframe)

    if standardize:
        plus_di = min_max_scale(plus_di, (0, 100)).round(8)
        list_to_std.add("plus_dm")

    dataframe["plus_dm"] = plus_dm
    dataframe["plus_di"] = plus_di
    return dataframe


def add_m_dim(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
    minus_dm = ta.MINUS_DM(dataframe)
    minus_di = ta.MINUS_DI(dataframe)

    if standardize:
        minus_di = min_max_scale(minus_di, (0, 100)).round(8)
        list_to_std.add("minus_dm")

    dataframe["minus_dm"] = minus_dm
    dataframe["minus_di"] = minus_di
    return dataframe


def add_aroon(data: pd.DataFrame, standardize: bool = False, *args, **kwargs):
    dataframe = data.copy()
    aroon = ta.AROON(dataframe)
    aroonup = aroon["aroonup"]
    aroondown = aroon["aroondown"]
    aroonosc = ta.AROONOSC(dataframe)

    if standardize:
        aroonup = min_max_scale(aroonup, (0, 100)).round(8)
        aroondown = min_max_scale(aroondown, (0, 100)).round(8)
        aroonosc = min_max_scale(aroonosc, (-100, 100), feature_range=(-1, 1)).round(8)

    dataframe["aroonup"] = aroonup
    dataframe["aroondown"] = aroondown
    dataframe["aroonosc"] = aroonosc
    return dataframe


def add_ao(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
    ao = i_tools.awesome_oscillator(dataframe)

    if standardize:
        list_to_std.add("ao")

    dataframe["ao"] = ao
    return dataframe


def add_keltner(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
    keltner = i_tools.keltner_channel(dataframe)
    keltner["percent"] = (dataframe["close"] - keltner["lower"]) / (
        keltner["upper"] - keltner["lower"]
    )
    keltner.loc[keltner.upper == keltner.lower, "percent"] = 0.5
    keltner["width"] = (
        keltner["upper"] - keltner["lower"]
    ) / keltner["mid"]

    if standardize:
        list_to_std.add("kc_upperband")
        list_to_std.add("kc_lowerband")
        list_to_std.add("kc_middleband")
        list_to_std.add("kc_percent")
        list_to_std.add("kc_width")

    dataframe["kc_upperband"] = keltner["upper"]
    dataframe["kc_lowerband"] = keltner["lower"]
    dataframe["kc_middleband"] = keltner["mid"]
    dataframe["kc_percent"] = keltner["percent"]
    dataframe["kc_width"] = keltner["width"]
    return dataframe


def add_uo(data: pd.DataFrame, standardize: bool = False, *args, **kwargs):
    dataframe = data.copy()
    uo = ta.ULTOSC(dataframe)

    if standardize:
        uo = min_max_scale(uo, (0, 100))

    dataframe["uo"] = uo
    return dataframe


def add_cci(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
    cci = ta.CCI(dataframe)

    if standardize:
        list_to_std.add("cci")

    dataframe["cci"] = cci
    return dataframe


def add_rsi(data: pd.DataFrame, standardize: bool = False, *args, **kwargs):
    dataframe = data.copy()
    rsi = ta.RSI(dataframe)

    if standardize:
        rsi = min_max_scale(rsi, (0, 100))

    dataframe["rsi"] = rsi
    return dataframe


def add_fisher_rsi(data: pd.DataFrame, *args, **kwargs):
    dataframe = data.copy()
    rsi = 0.1 * (ta.RSI(dataframe) - 50)
    dataframe["fisher_rsi"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
    return dataframe


def add_stochastic_slow(data: pd.DataFrame, standardize: bool = False, *args, **kwargs):
    dataframe = data.copy()
    stoch = ta.STOCH(dataframe)

    if standardize:
        stoch["slowd"] = min_max_scale(stoch["slowd"], (0, 100)).round(8)
        stoch["slowk"] = min_max_scale(stoch["slowk"], (0, 100)).round(8)

    dataframe["slowd"] = stoch["slowd"]
    dataframe["slowk"] = stoch["slowk"]
    return dataframe


def add_stochastic_fast(data: pd.DataFrame, standardize: bool = False, *args, **kwargs):
    dataframe = data.copy()
    stoch_fast = ta.STOCHF(dataframe)

    if standardize:
        stoch_fast["fastd"] = min_max_scale(stoch_fast["fastd"], (0, 100)).round(8)
        stoch_fast["fastk"] = min_max_scale(stoch_fast["fastk"], (0, 100)).round(8)

    dataframe["fastd"] = stoch_fast["fastd"]
    dataframe["fastk"] = stoch_fast["fastk"]
    return dataframe


def add_stoch_rsi(data: pd.DataFrame, standardize: bool = False, *args, **kwargs):
    dataframe = data.copy()
    stoch_rsi = ta.STOCHRSI(dataframe)

    if standardize:
        stoch_rsi["fastd"] = min_max_scale(stoch_rsi["fastd"], (0, 100)).round(8)
        stoch_rsi["fastk"] = min_max_scale(stoch_rsi["fastk"], (0, 100)).round(8)

    dataframe["fastd_rsi"] = stoch_rsi["fastd"]
    dataframe["fastk_rsi"] = stoch_rsi["fastk"]
    return dataframe


def add_macd(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
    macd = ta.MACD(dataframe)

    if standardize:
        list_to_std.add("macd")
        list_to_std.add("macdsignal")
        list_to_std.add("macdhist")

    dataframe["macd"] = macd["macd"]
    dataframe["macdsignal"] = macd["macdsignal"]
    dataframe["macdhist"] = macd["macdhist"]
    return dataframe


def add_mfi(data: pd.DataFrame, standardize: bool = False, *args, **kwargs):
    dataframe = data.copy()
    mfi = ta.MFI(dataframe)

    if standardize:
        mfi = min_max_scale(mfi, (0, 100)).round(8)

    dataframe["mfi"] = mfi
    return dataframe


def add_roc(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
    roc = ta.ROC(dataframe)

    if standardize:
        list_to_std.add("roc")

    dataframe["roc"] = roc
    return dataframe


def add_bbands(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
    bollinger = i_tools.bollinger_bands(
        i_tools.typical_price(dataframe), window=20, stds=2
    )
    bollinger["percent"] = (dataframe["close"] - bollinger["lower"]) / (
        bollinger["upper"] - bollinger["lower"]
    )
    bollinger.loc[bollinger.upper == bollinger.lower, "percent"] = 0.5
    bollinger["width"] = (
        bollinger["upper"] - bollinger["lower"]
    ) / bollinger["mid"]

    if standardize:
        list_to_std.add("bb_lowerband")
        list_to_std.add("bb_middleband")
        list_to_std.add("bb_upperband")
        list_to_std.add("bb_percent")
        list_to_std.add("bb_width")

    dataframe["bb_lowerband"] = bollinger["lower"]
    dataframe["bb_middleband"] = bollinger["mid"]
    dataframe["bb_upperband"] = bollinger["upper"]
    dataframe["bb_percent"] = bollinger["percent"]
    dataframe["bb_width"] = bollinger["width"]
    return dataframe


def add_wbbands(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
    weighted_bollinger = i_tools.weighted_bollinger_bands(
        i_tools.typical_price(dataframe), window=20, stds=2
    )
    weighted_bollinger["percent"] = (dataframe["close"] - weighted_bollinger["lower"]) / (
        weighted_bollinger["upper"] - weighted_bollinger["lower"]
    )
    weighted_bollinger.loc[weighted_bollinger.upper == weighted_bollinger.lower, "percent"] = 0.5
    weighted_bollinger["width"] = (
        weighted_bollinger["upper"] - weighted_bollinger["lower"]
    ) / weighted_bollinger["mid"]

    if standardize:
        list_to_std.add("wbb_upperband")
        list_to_std.add("wbb_lowerband")
        list_to_std.add("wbb_middleband")
        list_to_std.add("wbb_percent")
        list_to_std.add("wbb_width")

    dataframe["wbb_upperband"] = weighted_bollinger["upper"]
    dataframe["wbb_lowerband"] = weighted_bollinger["lower"]
    dataframe["wbb_middleband"] = weighted_bollinger["mid"]
    dataframe["wbb_percent"] = weighted_bollinger["percent"]
    dataframe["wbb_width"] = weighted_bollinger["width"]
    return dataframe


def add_ema(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
    dataframe["ema3"] = ta.EMA(dataframe, timeperiod=3)
    dataframe["ema5"] = ta.EMA(dataframe, timeperiod=5)
    dataframe["ema10"] = ta.EMA(dataframe, timeperiod=10)
    dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
    dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
    dataframe["ema100"] = ta.EMA(dataframe, timeperiod=100)

    if standardize:
        list_to_std.add("ema3")
        list_to_std.add("ema5")
        list_to_std.add("ema10")
        list_to_std.add("ema21")
        list_to_std.add("ema50")
        list_to_std.add("ema100")

    return dataframe


def add_sma(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
    dataframe["sma3"] = ta.SMA(dataframe, timeperiod=3)
    dataframe["sma5"] = ta.SMA(dataframe, timeperiod=5)
    dataframe["sma10"] = ta.SMA(dataframe, timeperiod=10)
    dataframe["sma21"] = ta.SMA(dataframe, timeperiod=21)
    dataframe["sma50"] = ta.SMA(dataframe, timeperiod=50)
    dataframe["sma100"] = ta.SMA(dataframe, timeperiod=100)

    if standardize:
        list_to_std.add("sma3")
        list_to_std.add("sma5")
        list_to_std.add("sma10")
        list_to_std.add("sma21")
        list_to_std.add("sma50")
        list_to_std.add("sma100")

    return dataframe


def add_sar(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()

    if standardize:
        list_to_std.add("sar")

    dataframe["sar"] = ta.SAR(dataframe)
    return dataframe


def add_tema(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
    dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

    if standardize:
        list_to_std.add("tema")

    return dataframe


def add_hilbert(data: pd.DataFrame, *args, **kwargs):
    dataframe = data.copy()
    hilbert = ta.HT_SINE(dataframe)
    dataframe["htsine"] = hilbert["sine"]
    dataframe["htleadsine"] = hilbert["leadsine"]
    return dataframe


def add_heikinashi(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
    heikinashi = i_tools.heikinashi(dataframe)
    dataframe["ha_open"] = heikinashi["open"]
    dataframe["ha_close"] = heikinashi["close"]
    dataframe["ha_high"] = heikinashi["high"]
    dataframe["ha_low"] = heikinashi["low"]

    if standardize:
        list_to_std.add("ha_open")
        list_to_std.add("ha_close")
        list_to_std.add("ha_high")
        list_to_std.add("ha_low")

    return dataframe


def add_ichimoku(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
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

    if standardize:
        list_to_std.add("ich_tenkan")
        list_to_std.add("ich_kijun")
        list_to_std.add("ich_spanA")
        list_to_std.add("ich_spanB")
        list_to_std.add("ich_chikou")
        list_to_std.add("ich_lead_spanA")
        list_to_std.add("ich_lead_spanB")

    return dataframe


def add_pattern_recognition_indicators(data: pd.DataFrame, *args, **kwargs):
    dataframe = data.copy()

    dataframe["CDLHAMMER"] = ta.CDLHAMMER(dataframe) / 100
    dataframe["CDLINVERTEDHAMMER"] = ta.CDLINVERTEDHAMMER(dataframe) / 100
    dataframe["CDLDRAGONFLYDOJI"] = ta.CDLDRAGONFLYDOJI(dataframe) / 100
    dataframe["CDLPIERCING"] = ta.CDLPIERCING(dataframe) / 100
    dataframe["CDLMORNINGSTAR"] = ta.CDLMORNINGSTAR(dataframe) / 100
    dataframe["CDL3WHITESOLDIERS"] = ta.CDL3WHITESOLDIERS(dataframe) / 100
    dataframe["CDLHANGINGMAN"] = ta.CDLHANGINGMAN(dataframe) / 100
    dataframe["CDLSHOOTINGSTAR"] = ta.CDLSHOOTINGSTAR(dataframe) / 100
    dataframe["CDLGRAVESTONEDOJI"] = ta.CDLGRAVESTONEDOJI(dataframe) / 100
    dataframe["CDLDARKCLOUDCOVER"] = ta.CDLDARKCLOUDCOVER(dataframe) / 100
    dataframe["CDLEVENINGDOJISTAR"] = ta.CDLEVENINGDOJISTAR(dataframe) / 100
    dataframe["CDLEVENINGSTAR"] = ta.CDLEVENINGSTAR(dataframe) / 100
    dataframe["CDL3LINESTRIKE"] = ta.CDL3LINESTRIKE(dataframe) / 100
    dataframe["CDLSPINNINGTOP"] = ta.CDLSPINNINGTOP(dataframe) / 100
    dataframe["CDLENGULFING"] = ta.CDLENGULFING(dataframe) / 100
    dataframe["CDLHARAMI"] = ta.CDLHARAMI(dataframe) / 100
    dataframe["CDL3OUTSIDE"] = ta.CDL3OUTSIDE(dataframe) / 100
    dataframe["CDL3INSIDE"] = ta.CDL3INSIDE(dataframe) / 100  # values [0, -100, 100]

    return dataframe


def add_candle_dir(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
    candle_dir = candle_direction(dataframe)

    if standardize:
        list_to_std.add("candle_dir")

    dataframe["candle_dir"] = candle_dir
    return dataframe


def add_candle_heigth(data: pd.DataFrame, standardize: bool = False, list_to_std: Set[str] = set()):
    dataframe = data.copy()
    candle_h = candle_height(dataframe)
    if standardize:
        list_to_std.add("candle_height")

    dataframe["candle_height"] = candle_h
    return dataframe


def add_percent_growth(
    data: pd.DataFrame,
    list_period: List[int] = [1, 3, 5],
    list_column: List[str] = ["open", "close", "low", "high"],
    standardize: bool = False,
    list_to_std: Set[str] = set()
):
    dataframe = data.copy()
    for period in list_period:
        new_name_column = list(map(lambda x: f"d{x}{period}", list_column))
        dataframe[new_name_column] = dataframe[list_column].pct_change(period)
        if standardize:
            for col in new_name_column:
                list_to_std.add(col)
    return dataframe


def add_SMA1(
    data: pd.DataFrame,
    standardize: bool = False,
    list_to_std: Set[str] = set()
):
    dataframe = data.copy()
    dataframe['SMA1'] = (dataframe.high + dataframe.low) / 2
    if standardize:
        list_to_std.add("SMA1")
    return dataframe


def add_ln_SMA1(
    data: pd.DataFrame,
    standardize: bool = False,
    list_to_std: Set[str] = set()
):
    drop_SMA1 = False
    if 'SMA1' not in data:
        dataframe = add_SMA1(data.copy())
        drop_SMA1 = True
    else:
        dataframe = data.copy()

    dataframe['lnSMA1'] = np.log(dataframe['SMA1'])

    if drop_SMA1:
        dataframe.drop('SMA1', axis=1, inplace=True)

    if standardize:
        list_to_std.add("lnSMA1")
    return dataframe


def add_return(
    data: pd.DataFrame,
    offset: int = 1,
    colname: str = "return",
    dropna: bool = False,
    standardize: bool = False,
    list_to_std: Set[str] = set()
):
    dataframe = data.copy()
    dataframe[colname] = dataframe.close / dataframe.shift(offset).close
    dataframe.dropna(inplace=dropna)
    if standardize:
        list_to_std.add(colname)
    return dataframe


def add_stats_indicators_on_rolling_windows(
    data: pd.DataFrame,
    standardize: bool = False,
    list_to_std: Set[str] = set()
):
    dataframe = data.copy()
    dataframe = add_stats_indicators(
        dataframe, 3, standardize=standardize, list_to_std=list_to_std)
    dataframe = add_stats_indicators(
        dataframe, 5, standardize=standardize, list_to_std=list_to_std)
    dataframe = add_stats_indicators(
        dataframe, 10, standardize=standardize, list_to_std=list_to_std)
    dataframe = add_stats_indicators(
        dataframe, 21, standardize=standardize, list_to_std=list_to_std)
    dataframe = add_stats_indicators(
        dataframe, 50, standardize=standardize, list_to_std=list_to_std)
    dataframe = add_stats_indicators(
        dataframe, 100, standardize=standardize, list_to_std=list_to_std)
    return dataframe


def add_stats_indicators(
    data: pd.DataFrame,
    rolling_window: int = 1,
    standardize: bool = False,
    list_to_std: Set[str] = set()
):
    dataframe = data.copy()
    dataframe = add_SMA1(dataframe, standardize, list_to_std)
    if rolling_window >= 4:
        dataframe[f"kurtosis{rolling_window}"] = kurtosis(dataframe, "SMA1", rolling_window)
    dataframe[f"skewness{rolling_window}"] = skewness(dataframe, "SMA1", rolling_window)
    dataframe[f"median{rolling_window}"] = median(dataframe, "SMA1", rolling_window)
    dataframe[f"minimum{rolling_window}"] = minimum(dataframe, "SMA1", rolling_window)
    dataframe[f"maximum{rolling_window}"] = maximum(dataframe, "SMA1", rolling_window)
    dataframe[f"std{rolling_window}"] = standard_deviation(dataframe, "SMA1", rolling_window)
    dataframe[f"variance{rolling_window}"] = variance(dataframe, "SMA1", rolling_window)
    dataframe[f"quantile025_{rolling_window}"] = quantile_025(dataframe, "SMA1", rolling_window)
    dataframe[f"quantile075_{rolling_window}"] = quantile_075(dataframe, "SMA1", rolling_window)
    dataframe[f"mean{rolling_window}"] = mean(dataframe, "SMA1", rolling_window)

    if standardize:
        if f"kurtosis{rolling_window}" in list(dataframe.columns):
            list_to_std.add(f"kurtosis{rolling_window}")
        list_to_std.add(f"skewness{rolling_window}")
        list_to_std.add(f"median{rolling_window}")
        list_to_std.add(f"minimum{rolling_window}")
        list_to_std.add(f"maximum{rolling_window}")
        list_to_std.add(f"std{rolling_window}")
        list_to_std.add(f"variance{rolling_window}")
        list_to_std.add(f"quantile025_{rolling_window}")
        list_to_std.add(f"quantile075_{rolling_window}")
        list_to_std.add(f"mean{rolling_window}")

    return dataframe


def candle_direction(data: pd.DataFrame):
    dataframe = data.copy()
    return np.log(dataframe.close) - np.log(dataframe.open)


def candle_height(data: pd.DataFrame):
    dataframe = data.copy()
    return np.log(dataframe.high) - np.log(dataframe.low)


def kurtosis(data: pd.DataFrame, column_name: str = "close", rolling_window: int = 1):
    dataframe = data.copy()
    if rolling_window < 4:
        raise Exception("Kurtosis should have a rolling window >= 4")
    return dataframe[column_name].rolling(rolling_window).kurt()


def skewness(data: pd.DataFrame, column_name: str = "close", rolling_window: int = 1):
    dataframe = data.copy()
    return dataframe[column_name].rolling(rolling_window).skew()


def median(data: pd.DataFrame, column_name: str = "close", rolling_window: int = 1):
    dataframe = data.copy()
    return dataframe[column_name].rolling(rolling_window).median()


def minimum(data: pd.DataFrame, column_name: str = "close", rolling_window: int = 1):
    dataframe = data.copy()
    return dataframe[column_name].rolling(rolling_window).min()


def maximum(data: pd.DataFrame, column_name: str = "close", rolling_window: int = 1):
    dataframe = data.copy()
    return dataframe[column_name].rolling(rolling_window).max()


def standard_deviation(data: pd.DataFrame, column_name: str = "close", rolling_window: int = 1):
    dataframe = data.copy()
    return dataframe[column_name].rolling(rolling_window).std()


def variance(data: pd.DataFrame, column_name: str = "close", rolling_window: int = 1):
    dataframe = data.copy()
    return dataframe[column_name].rolling(rolling_window).var()


def volatility(data: pd.DataFrame, column_name: str = "close", rolling_window: int = 1):
    dataframe = data.copy()
    return dataframe[column_name].rolling(rolling_window).var()


def quantile_025(data: pd.DataFrame, column_name: str = "close", rolling_window: int = 1):
    dataframe = data.copy()
    return dataframe[column_name].rolling(rolling_window).quantile(0.25)


def quantile_075(data: pd.DataFrame, column_name: str = "close", rolling_window: int = 1):
    dataframe = data.copy()
    return dataframe[column_name].rolling(rolling_window).quantile(0.75)


def mean(data: pd.DataFrame, column_name: str = "close", rolling_window: int = 1):
    dataframe = data.copy()
    return dataframe[column_name].rolling(rolling_window).mean()
