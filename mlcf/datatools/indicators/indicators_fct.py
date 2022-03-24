"""Indicators function mododule.
Provide a set of indicators functions.
"""

from typing import Callable, Dict
# import mlcf.datatools.indicators_tools as i_tools
# import numpy as np
import pandas as pd
import pandas_ta as _  # noqa
import talib.abstract as ta


__all__ = [
    "add_adx",
    "INDICE_DICT"
]


def add_adx(data: pd.DataFrame, *args, **kwargs):
    """Average Directional Index (ADX)
    The average directional index (ADX) is a technical analysis indicator used by some traders
    to determine the strength of a trend.
    NOTE: The ADX function has an unstable period.
    real = ADX(high, low, close, timeperiod=14)
    """
    dataframe = data.copy()
    dataframe["adx"] = ta.ADX(data, *args, **kwargs)
    return dataframe


INDICE_DICT: Dict[str, Callable] = {
    "adx": add_adx
}
