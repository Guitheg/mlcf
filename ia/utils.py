# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from typing import Callable, List

def dataframe_to_numpy(data_df: pd.DataFrame,
                       columns: List[str]) -> np.ndarray:
    data = np.array(data_df[columns])
        
    return data

def _to_pd(fct: Callable, 
           data_df: pd.DataFrame, 
           columns: List[str], *args, **kwargs) -> pd.DataFrame:
    data = dataframe_to_numpy(data_df=data_df, columns=columns)
    data = fct(data, *args, **kwargs)
    new_data_df = pd.DataFrame(data=data, columns=columns)
    for col in set(data_df.columns) ^ set(columns):
        new_data_df[col] = data_df[col]
    return new_data_df

def standardize_data(data: np.ndarray,
                     with_std_mean: bool = True, 
                     with_std_scale: bool = True) -> np.ndarray:
    data = StandardScaler(with_mean = with_std_mean, with_std = with_std_scale).fit_transform(data)
    return data


def standardize_data_df(data_df: pd.DataFrame, 
                        columns: List[str], 
                        *args, **kwargs) -> pd.DataFrame:
    return _to_pd(standardize_data, data_df, columns, *args, **kwargs)


def stationnarize_data_df(data_df: pd.DataFrame, 
                          columns: List[str], 
                          *args, **kwargs) -> pd.DataFrame:
    return _to_pd(stationnarize_data, data_df, columns, *args, **kwargs)


def untrend_data_df(data_df: pd.DataFrame, 
                    columns: List[str], 
                    *args, **kwargs) -> pd.DataFrame:
    return _to_pd(untrend_data, data_df, columns, *args, **kwargs)


def stationnarize_data(data: np.ndarray) -> np.ndarray:
    return (data - np.roll(data, 1, axis=0)) [1:]


def untrend_data(data: np.ndarray) -> np.ndarray:
    _,c = data.shape
    data_trend = np.ndarray(data.shape)
    list_coef = []
    for i in range(c):
        trend, (a, b) = data_trend_one_axis(data, i)
        data_trend[:,i] = trend
        list_coef.append((a,b))
    return data-data_trend


def data_trend_one_axis(data: np.ndarray, num_axis: int = 3) -> np.ndarray:
    linear_regression = LinearRegression().fit(np.arange(len(data)).reshape(-1, 1), 
                                               data[:,num_axis])
    data_trend = (np.arange(len(data))*linear_regression.coef_)+linear_regression.intercept_
    return data_trend, (linear_regression.coef_, linear_regression.intercept_)

def add_past_shifted_columns(dataframe, list_columns, n_shift):
    data = dataframe.copy()
    for i in range(1, n_shift):
        for col in list_columns:
            data[f"{col}{i}"] = data[col].shift(i)
    return data

