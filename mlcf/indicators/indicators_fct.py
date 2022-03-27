"""Indicators Function mododule.

Provide a set of indicators functions and INDICE_DICT which give one of these function given a
string key.
"""

from typing import Callable, Dict
# import mlcf.indicators_tools as i_tools
# import numpy as np
import pandas as pd
import pandas_ta as _  # noqa
import talib.abstract as ta

# TODO (doc) correct English

__all__ = [
    "add_adx",
    "indice_dict"
]


def add_adx(data: pd.DataFrame, *args, **kwargs):
    """Average Directional Index (ADX)
    The average directional index (ADX) is a technical analysis indicator used by some traders
    to determine the strength of a trend.
    note: The ADX function has an unstable period.
    real = ADX(high, low, close, timeperiod=14)
    """
    dataframe = data.copy()
    dataframe["adx"] = ta.ADX(data, *args, **kwargs)
    return dataframe


def indice_dict(indice_name: str) -> Callable:
    """From an indicator name it returns the corresponding indicator function.

    Indicators available:

    Attributes:
        {list_indice}

    Args:
        indice_name (str): An indicator name.

    Returns:
        Callable: The corresponding function.
    """
    return _INDICE_DICT[indice_name]


_INDICE_DICT: Dict[str, Callable] = {
    "adx": add_adx
}
indice_dict.__doc__ = str(indice_dict.__doc__).format(list_indice=list(_INDICE_DICT.keys()))
