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
    
    if n_components == 2:
        fig = plotly.subplots.make_subplots(rows=3, cols=2)
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
        print(pca.noise_variance_)
        print(pca.get_covariance())
        df_components = pd.DataFrame(data=pca.components_.T, columns=['pc1', 'pc2'])
        print(df_components)

        compo = px.scatter(df_components, x="pc1", y="pc2")["data"][0]
        fig.add_trace(
            compo,
            row=3,
            col=2
        )
        
        hist = px.histogram(y=pca.explained_variance_ratio_, x=np.arange(len(pca.explained_variance_ratio_)))["data"][0]
        fig.add_trace(
            hist,
            row=1,
            col=2
        )
        
        hist = px.histogram(y=pca.singular_values_, x=np.arange(len(pca.explained_variance_ratio_)))["data"][0]
        fig.add_trace(
            hist,
            row=2,
            col=2
        )
        
        if plot == True:
            fig.show()
    else:
        raise NotImplementedError("n_component =/= 2 not implemented")

    return pcd_data_df_result, (trace[0], theline[0])


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


def binary_soft_search(x, list: List, low = 0, high = -1):
    high = len(list)-1 if high == -1 else high
    if high <= low:
        if low <= 0 :
            return 0
        else:
            return len(list)-1
    mid = int((low + high) / 2)
    if x == list[mid] or x > list[mid] and x < list[mid+1]:
        return mid
    if x < list[mid]:
        return binary_soft_search(x, list, low, mid-1)
    if x > list[mid]:
        return binary_soft_search(x, list, mid+1, high)



def analyse_acp(data_df: pd.DataFrame, 
                columns : List[str], 
                standardize : bool = True, 
                mode : str = "default",
                name : str = None):
    
    data = data_df[columns].copy()
    
    if mode == "untrend":
        data = untrend_data_df(data, columns)
    
    elif mode == "stationarize":
        data = stationnarize_data_df(data, columns)
        
    if standardize:
        data = standardize_data_df(data, columns)

    acp = PCA()
    
    acp_results = acp.fit_transform(data)
    acp_results_df = pd.DataFrame(data=acp_results, 
                                    columns=[f"F{i+1}" for i in range(acp.n_components_)])

    explained_variance_ratio = acp.explained_variance_ratio_
    
    fig = plotly.subplots.make_subplots(rows=2, cols=3,
                                        subplot_titles=(
                                            'Explained Variance Ratio', 
                                            'Values represented on the Component Plan', 
                                            'Values Contribution in the Total Inertie',
                                            'title 4',
                                            "Données d'entrée"))
    fig.update_layout(
        title_text=f"Analyse en Composantes Principales (ACP)"+\
        f"{(f' - {name}' if not name is None else '')} - Colonnes étudiées : {columns}")

    # First Plot ###
    fig.add_trace(px.bar(explained_variance_ratio)["data"][0],
                  row=1, col=1)
    ####
    
    # Second Plot ###
    min_val = min(np.min(acp_results_df[acp_results_df.columns[0]]), 
                        np.min(acp_results_df[acp_results_df.columns[1]]))
    max_val = max(np.max(acp_results_df[acp_results_df.columns[0]]), 
                        np.max(acp_results_df[acp_results_df.columns[1]]))
    acp_results_trace = px.scatter(acp_results_df,
                       x = acp_results_df.columns[0],
                       y = acp_results_df.columns[1])["data"][0]
    acp_results_trace["mode"] = 'markers'
    acp_results_trace["marker"]["symbol"] = "circle"
    acp_results_trace["marker"]["size"] = 3

    fig.add_trace(acp_results_trace,
                  row=1, col=2)
    fig.update_layout(xaxis2 = dict(range=[min_val, max_val]))
    fig.update_layout(yaxis2 = dict(range=[min_val, max_val]))
    ####
    
    total_inertie = (data**2).sum(axis=1)
    
    # Third Plot ###
    fig.add_trace(px.line(total_inertie)["data"][0],
                  row=1, col=3)
    ####
    
    # fig.add_trace(,
    #               row=2, col=1)
    
    # Fith plot ###
    fig.add_trace(px.line(data[data.columns[0]])["data"][0],
                  row=2, col=2)
    ####
    # fig.add_trace(,
    #               row=2, col=3)

    fig.show()
    

def main():
    print(os.path.abspath(os.path.curdir))
    BTC_BUSD_1m = load_pair_history("BTC/BUSD", "1m", Path("./user_data/data/binance"))
    # print(BTC_BUSD_1m)
    
    analyse_acp(BTC_BUSD_1m, COL, name="Cours du BTC", mode = "default")
    
    data = dataframe_to_numpy(BTC_BUSD_1m, columns=COL)
    
    # data = standardize_data(data)
    # data_df = standardize_data_df(BTC_BUSD_1m, columns=COL)

    # untrended_data = untrend_data(data)
    # untrended_data_df = untrend_data_df(data_df, columns=COL)
    
    # stationarized_data = stationnarize_data(data)
    # stationarized_data_df = stationnarize_data_df(data_df, columns=COL)
    
    # data_trend,_ = data_trend_one_axis(data, 3)

    # acp = PCA()
    # r = acp.fit_transform(stationarized_data_df[COL])
    # hist = px.bar(y=acp.explained_variance_ratio_, x=np.arange(len(acp.explained_variance_ratio_)), title="explained variance ratio")
    # hist.show()
    # hist = px.bar(y=acp.singular_values_, x=np.arange(len(acp.singular_values_)), title="singular_values")
    # hist.show()
    # pca_result_untrended, (pca1, line1) = process_pca(untrended_data_df, n_components=2, plot=True)
    # pca_result_stationarized, (pca2, line2) = process_pca(stationarized_data_data_df, n_components=2, plot=True)




    # histogram
    # hist_varia = px.histogram(stationarized_data_data_df, x="close", title="ratio variation histogram")
    # hist_cov = px.histogram()
    # print(hist_varia)
    
    # fig = plotly.subplots.make_subplots(rows=4, cols=2)
    # fig.add_trace(hist_varia['data'][0],  row=4, col=1)
    # fig.add_trace(go.Scatter(x=np.arange(len(data)), y=data[:,3]), row=1, col=1)
    # fig.add_trace(go.Scatter(x=np.arange(len(data)), y=data_trend, mode="lines"), row=1, col=1)
    # fig.add_trace(line2, row=2, col=1)
    # fig.add_trace(line1, row=3, col=1)
    
    # fig.add_trace(pca1,  row=3, col=2)
    # fig.add_trace(pca2,  row=2, col=2)
    # fig.show()
    
    # process_pca(untrended_data_df, n_components=2, plot=True)
    # process_pca(stationarized_data_df, n_components=2, plot=True)

    pass
    
if __name__ == "__main__":
    main()