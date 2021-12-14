import os
from re import T
from typing import Callable, List
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.decomposition import PCA
from freqtrade.data.history.history_utils import load_pair_history
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


COL = ["open", "high", "low", "close", "volume"]


def dataframe_to_numpy(data_df: pd.DataFrame,
                       columns: List[str]) -> np.ndarray:
    data = np.array(data_df[columns])
        
    return data
    
def process_pca(data_df: pd.DataFrame, n_components: int, plot: bool = False) -> pd.DataFrame:
    data = dataframe_to_numpy(data_df, columns=COL)
    data = standardize_data(data)

    pca = PCA(n_components=n_components)
    pca_data_result = pca.fit_transform(data)
    pcd_data_df_result = pd.DataFrame(data=pca_data_result, 
                                      columns=[f"C{i+1}" for i in range(n_components)])
    pcd_data_df_result["date"] = data_df["date"]
    pcd_data_df_result["color"] = np.linspace(0, 255, len(pcd_data_df_result))
    if plot == True:
        if n_components == 2:
            fig = plotly.subplots.make_subplots(rows=2, cols=1)
            trace = px.scatter(pcd_data_df_result,
                                         text = 'date',
                                         x = pcd_data_df_result.columns[0],
                                         y = pcd_data_df_result.columns[1],
                                         color = 'color')["data"]
            # trace[0]["mode"] = 'lines'
            trace[0]["mode"] = 'markers'
            trace[0]["marker"]["symbol"] = "circle"
            trace[0]["marker"]["size"] = 3
            fig.add_trace(
                trace[0],
                row=1,
                col=1
            )
            data_df["color"] = pcd_data_df_result["color"]

            theline = px.scatter(data_df, x="date", y="close", color="color")["data"]
            theline[0]["mode"] = "markers"
            fig.add_trace(
                theline[0],
                row=2,
                col=1
            )
            fig.show()
        else:
            raise NotImplementedError("n_component =/= 2 not implemented")
    
    return pcd_data_df_result


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


def data_trend_one_axis(data: np.ndarray, axis = 3) -> np.ndarray:
    linear_regression = LinearRegression().fit(np.arange(len(data)).reshape(-1, 1), data[:,axis])
    data_trend = (np.arange(len(data))*linear_regression.coef_)+linear_regression.intercept_
    return data_trend, (linear_regression.coef_, linear_regression.intercept_)


def main():
    print(os.path.abspath(os.path.curdir))
    BTC_BUSD_1m = load_pair_history("BTC/BUSD", "1m", Path("./user_data/data/binance"))
    
    # pca_result = process_pca(BTC_BUSD_5m, n_components=2, plot=True)
    data = dataframe_to_numpy(BTC_BUSD_1m, columns=COL)
    
    data = standardize_data(data)
    data_df = standardize_data_df(BTC_BUSD_1m, columns=COL)

    untrended_data = untrend_data(data)
    untrended_data_df = untrend_data_df(data_df, columns=COL)
    
    no_stationary_data = stationnarize_data(data)
    no_stationary_data_df = stationnarize_data_df(data_df, columns=COL)
    
    data_trend,_ = data_trend_one_axis(data, 3)

    fig = plotly.subplots.make_subplots(rows=3, cols=1)
    fig.add_trace(go.Scatter(x=np.arange(len(data)), y=data[:,3]), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(data)), y=data_trend, mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(no_stationary_data)), y=no_stationary_data[:,3]), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(untrended_data)), y=untrended_data[:,3]), row=3, col=1)

    fig.show()
    
    process_pca(untrended_data_df, n_components=2, plot=True)
    process_pca(no_stationary_data_df, n_components=2, plot=True)

    pass
    
if __name__ == "__main__":
    main()