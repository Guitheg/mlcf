"""Custom Indicators Module


"""

from mlcf.indicators import _indicators_tools as i_tools
import pandas as pd
import pandas_ta as _  # noqa
from typing import Callable, Dict

__all__ = [
    "CUSTOM_FEATURES",
    "add_custom_feature",
    "crossed",
    "crossed_above",
    "crossed_below",
    "hma",
    "returns",
    "log_returns",
    "implied_volatility",
    "zlma",
    "pct_change"
]


def crossed(data: pd.DataFrame, column_a: str, column_b: str) -> pd.DataFrame:
    dataframe = data.copy()
    return i_tools.crossed(dataframe[column_a], dataframe[column_b])


def crossed_above(data: pd.DataFrame, column_a: str, column_b: str) -> pd.DataFrame:
    dataframe = data.copy()
    return i_tools.crossed_above(dataframe[column_a], dataframe[column_b])


def crossed_below(data: pd.DataFrame, column_a: str, column_b: str) -> pd.DataFrame:
    dataframe = data.copy()
    return i_tools.crossed_below(dataframe[column_a], dataframe[column_b])


def hma(data: pd.DataFrame, column: str, *args, **kwargs):
    dataframe = data.copy()
    return i_tools.hull_moving_average(dataframe[column], *args, **kwargs)


def returns(data: pd.DataFrame, column: str, *args, **kwargs):
    dataframe = data.copy()
    return i_tools.returns(dataframe[column], *args, **kwargs)


def log_returns(data: pd.DataFrame, column: str, *args, **kwargs):
    dataframe = data.copy()
    return i_tools.log_returns(dataframe[column], *args, **kwargs)


def implied_volatility(data: pd.DataFrame, column: str, *args, **kwargs):
    dataframe = data.copy()
    return i_tools.implied_volatility(dataframe[column], *args, **kwargs)


def zlma(data: pd.DataFrame, column: str, *args, **kwargs):
    dataframe = data.copy()
    return i_tools.zlma(dataframe[column], *args, **kwargs)


def pct_change(data: pd.DataFrame, column: str, timeperiod: int = 1):
    dataframe = data.copy()
    return dataframe[column].pct_change(timeperiod)


CUSTOM_FEATURES: Dict[str, Callable] = {
    "crossed": crossed,
    "crossed_above": crossed_above,
    "crossed_below": crossed_below,
    "hma": hma,
    "returns": returns,
    "log_returns": log_returns,
    "implied_volatility": implied_volatility,
    "zlma": zlma,
    "pct_change": pct_change
}


def add_custom_feature(
    data: pd.DataFrame,
    custom_feature_name: str,
    custom_column_name: str = None,
    *args,
    **kwargs
) -> pd.DataFrame:
    """Add a custom feature to the dataframe.

    Args:
        data (pd.DataFrame): The dataframe

        custom_feature_name (str): The custom feature name. Please choose among this :
            {feature_list}

        custom_column_name (str, optional): The custom name for the new created column.
            Default to None.

    Returns:
        pd.DataFrame: The dataframe with the added feature
    """
    dataframe = data.copy()
    results = pd.DataFrame(CUSTOM_FEATURES[custom_feature_name](dataframe, *args, **kwargs))

    param_string: str = "" if not len(kwargs) else f"{kwargs}"
    if custom_column_name is not None and isinstance(custom_column_name, str):
        new_column_name = custom_column_name
    else:
        new_column_name = custom_feature_name+str(param_string)
    if isinstance(results, pd.Series):
        results.name = new_column_name
    if len(results.columns) == 1:
        results.columns = [new_column_name]
    return pd.concat([dataframe, results], axis=1)


add_custom_feature.__doc__ = str(add_custom_feature.__doc__).format(
    feature_list=list(CUSTOM_FEATURES.keys())
)
