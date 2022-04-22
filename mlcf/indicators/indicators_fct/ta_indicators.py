"""Technical Analysis Indicators Module

See https://mrjbq7.github.io/ta-lib/funcs.html for more info.
"""

from mlcf.indicators import _indicators_tools as i_tools
import pandas as pd
import pandas_ta as _  # noqa
import talib.abstract as ta
from typing import Callable, Dict

__all__ = [
    "TA_FEATURES",
    "add_ta_feature"
]


TA_FEATURES: Dict[str, Callable] = {
    "adx": ta.ADX,
    "adxr": ta.ADXR,
    "apo": ta.APO,
    "aroon": ta.AROON,
    "aroonosc": ta.AROONOSC,
    "bop": ta.BOP,
    "cci": ta.CCI,
    "cmo": ta.CMO,
    "dx": ta.DX,
    "macd": ta.MACD,
    "macdext": ta.MACDEXT,
    "macdfix": ta.MACDFIX,
    "mfi": ta.MFI,
    "minus_di": ta.MINUS_DI,
    "minus_dm": ta.MINUS_DM,
    "mom": ta.MOM,
    "plus_di": ta.PLUS_DI,
    "plus_dm": ta.PLUS_DM,
    "ppo": ta.PPO,
    "roc": ta.ROC,
    "rocp": ta.ROCP,
    "rocr": ta.ROCR,
    "rocr100": ta.ROCR100,
    "rsi": ta.RSI,
    "stoch": ta.STOCH,
    "stochf": ta.STOCHF,
    "stochrsi": ta.STOCHRSI,
    "trix": ta.TRIX,
    "ultosc": ta.ULTOSC,
    "willr": ta.WILLR,
    "bbands": i_tools.bollinger_bands,
    "dema": ta.DEMA,
    "ema": ta.EMA,
    "ht_trendline": ta.HT_TRENDLINE,
    "kama": ta.KAMA,
    "ma": ta.MA,
    "mama": ta.MAMA,
    "mavp": ta.MAVP,
    "midpoint": ta.MIDPOINT,
    "midprice": ta.MIDPRICE,
    "sar": ta.SAR,
    "sarext": ta.SAREXT,
    "sma": ta.SMA,
    "t3": ta.T3,
    "tema": ta.TEMA,
    "trima": ta.TRIMA,
    "wma": ta.WMA,
    "ad": ta.AD,
    "adosc": ta.ADOSC,
    "obv": ta.OBV,
    "ht_dcperiod": ta.HT_DCPERIOD,
    "ht_dcphase": ta.HT_DCPHASE,
    "ht_phasor": ta.HT_PHASOR,
    "ht_sine": ta.HT_SINE,
    "ht_trendmode": ta.HT_TRENDMODE,
    "avgprice": ta.AVGPRICE,
    "medprice": ta.MEDPRICE,
    "typprice": ta.TYPPRICE,
    "wclprice": ta.WCLPRICE,
    "atr": ta.ATR,
    "natr": ta.NATR,
    "trange": ta.TRANGE,
    "cdl2crows": ta.CDL2CROWS,
    "cdl3blackcrows": ta.CDL3BLACKCROWS,
    "cdl3inside": ta.CDL3INSIDE,
    "cdl3linestrike": ta.CDL3LINESTRIKE,
    "cdl3outside": ta.CDL3OUTSIDE,
    "cdl3starsinsouth": ta.CDL3STARSINSOUTH,
    "cdl3whitesoldiers": ta.CDL3WHITESOLDIERS,
    "cdlabandonedbaby": ta.CDLABANDONEDBABY,
    "cdladvanceblock": ta.CDLADVANCEBLOCK,
    "cdlbelthold": ta.CDLBELTHOLD,
    "cdlbreakaway": ta.CDLBREAKAWAY,
    "cdlclosingmarubozu": ta.CDLCLOSINGMARUBOZU,
    "cdlconcealbabyswall": ta.CDLCONCEALBABYSWALL,
    "cdlcounterattack": ta.CDLCOUNTERATTACK,
    "cdldarkcloudcover": ta.CDLDARKCLOUDCOVER,
    "cdldoji": ta.CDLDOJI,
    "cdldojistar": ta.CDLDOJISTAR,
    "cdldragonflydoji": ta.CDLDRAGONFLYDOJI,
    "cdlengulfing": ta.CDLENGULFING,
    "cdleveningdojistar": ta.CDLEVENINGDOJISTAR,
    "cdleveningstar": ta.CDLEVENINGSTAR,
    "cdlgapsidesidewhite": ta.CDLGAPSIDESIDEWHITE,
    "cdlgravestonedoji": ta.CDLGRAVESTONEDOJI,
    "cdlhammer": ta.CDLHAMMER,
    "cdlhangingman": ta.CDLHANGINGMAN,
    "cdlharami": ta.CDLHARAMI,
    "cdlharamicross": ta.CDLHARAMICROSS,
    "cdlhighwave": ta.CDLHIGHWAVE,
    "cdlhikkake": ta.CDLHIKKAKE,
    "cdlhikkakemod": ta.CDLHIKKAKEMOD,
    "cdlhomingpigeon": ta.CDLHOMINGPIGEON,
    "cdlidentical3crows": ta.CDLIDENTICAL3CROWS,
    "cdlinneck": ta.CDLINNECK,
    "cdlinvertedhammer": ta.CDLINVERTEDHAMMER,
    "cdlkicking": ta.CDLKICKING,
    "cdlkickingbylength": ta.CDLKICKINGBYLENGTH,
    "cdlladderbottom": ta.CDLLADDERBOTTOM,
    "cdllongleggeddoji": ta.CDLLONGLEGGEDDOJI,
    "cdllongline": ta.CDLLONGLINE,
    "cdlmarubozu": ta.CDLMARUBOZU,
    "cdlmatchinglow": ta.CDLMATCHINGLOW,
    "cdlmathold": ta.CDLMATHOLD,
    "cdlmorningdojistar": ta.CDLMORNINGDOJISTAR,
    "cdlmorningstar": ta.CDLMORNINGSTAR,
    "cdlonneck": ta.CDLONNECK,
    "cdlpiercing": ta.CDLPIERCING,
    "cdlrickshawman": ta.CDLRICKSHAWMAN,
    "cdlrisefall3methods": ta.CDLRISEFALL3METHODS,
    "cdlseparatinglines": ta.CDLSEPARATINGLINES,
    "cdlshootingstar": ta.CDLSHOOTINGSTAR,
    "cdlshortline": ta.CDLSHORTLINE,
    "cdlspinningtop": ta.CDLSPINNINGTOP,
    "cdlstalledpattern": ta.CDLSTALLEDPATTERN,
    "cdlsticksandwich": ta.CDLSTICKSANDWICH,
    "cdltakuri": ta.CDLTAKURI,
    "cdltasukigap": ta.CDLTASUKIGAP,
    "cdlthrusting": ta.CDLTHRUSTING,
    "cdltristar": ta.CDLTRISTAR,
    "cdlunique3river": ta.CDLUNIQUE3RIVER,
    "cdlupsidegap2crows": ta.CDLUPSIDEGAP2CROWS,
    "cdlxsidegap3methods": ta.CDLXSIDEGAP3METHODS,
    "beta": ta.BETA,
    "correl": ta.CORREL,
    "linearreg": ta.LINEARREG,
    "linearreg_angle": ta.LINEARREG_ANGLE,
    "linearreg_intercept": ta.LINEARREG_INTERCEPT,
    "linearreg_slope": ta.LINEARREG_SLOPE,
    "stddev": ta.STDDEV,
    "tsf": ta.TSF,
    "var": ta.VAR,
    "acos": ta.ACOS,
    "asin": ta.ASIN,
    "atan": ta.ATAN,
    "ceil": ta.CEIL,
    "cos": ta.COS,
    "cosh": ta.COSH,
    "exp": ta.EXP,
    "floor": ta.FLOOR,
    "ln": ta.LN,
    "log10": ta.LOG10,
    "sin": ta.SIN,
    "sinh": ta.SINH,
    "sqrt": ta.SQRT,
    "tan": ta.TAN,
    "tanh": ta.TANH,
    "add": ta.ADD,
    "div": ta.DIV,
    "max": ta.MAX,
    "maxindex": ta.MAXINDEX,
    "min": ta.MIN,
    "minindex": ta.MININDEX,
    "minmax": ta.MINMAX,
    "minmaxindex": ta.MINMAXINDEX,
    "mult": ta.MULT,
    "sub": ta.SUB,
    "sum": ta.SUM,
    "heikinashi": i_tools.heikinashi,
    "tdi": i_tools.tdi,
    "ao": i_tools.awesome_oscillator,
    "t_price": i_tools.typical_price,
    "mid_price": i_tools.mid_price,
    "ibs": i_tools.ibs,
    "true_range": i_tools.true_range,
    "rvwap": i_tools.rolling_vwap,
    "wbbands": i_tools.weighted_bollinger_bands,
    "kc": i_tools.keltner_channel,
    "zscore": i_tools.zscore,
    "pvt": i_tools.pvt,
    "chopiness": i_tools.chopiness
}


def add_ta_feature(
    data: pd.DataFrame,
    ta_feature_name: str,
    custom_column_name: str = None,
    *args,
    **kwargs
) -> pd.DataFrame:
    """Add a technical analysis indicator in the dataframe.

    See https://mrjbq7.github.io/ta-lib/funcs.html for more info.

    Args:
        data (pd.DataFrame): The dataframe

        ta_feature_name (str): The name of the technical indicator. Please choose among this list:
            {feature_list}

        custom_column_name (str, optional): The custom name for the new created column.
            Default to None.

    Returns:
        pd.DataFrame: The dataframe with the added indicator.
    """
    dataframe = data.copy()
    results = pd.DataFrame(TA_FEATURES[ta_feature_name](dataframe, *args, **kwargs))

    param_string: str = "" if not len(kwargs) else f"{kwargs}"
    if custom_column_name is not None and isinstance(custom_column_name, str):
        new_column_name = custom_column_name
    else:
        new_column_name = ta_feature_name+str(param_string)
    if isinstance(results, pd.Series):
        results.name = new_column_name

    if len(results.columns) == 1:
        results = results.rename(columns={results.columns[0]: new_column_name})
    else:
        for col in results.columns:
            results = results.rename(columns={col: f"({col})"+new_column_name})

    return pd.concat([dataframe, results], axis=1)


add_ta_feature.__doc__ = str(add_ta_feature.__doc__).format(
    feature_list=list(TA_FEATURES.keys())
)
