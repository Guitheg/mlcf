# -*- coding: utf-8 -*-

import os
from re import T
from typing import List

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.decomposition import PCA
from freqtrade.data.history.history_utils import load_pair_history
from sklearn.cluster import KMeans
import argparse
from pandas_ta.overlap.ichimoku import ichimoku
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from utils import add_past_shifted_columns, untrend_data_df, standardize_data_df, \
    stationnarize_data_df

def analyse_acp(data_df: pd.DataFrame, 
                columns : List[str], 
                mode : str = "default",
                input_plot : str = None,
                name : str = None,
                n_cluster : int = 2,
                n_cpt_kmeans : int = 2,
                plot : bool = True) -> dict:
    data = data_df.copy()
    
    if input_plot is None:
        input_plot = columns[0]
    
    if mode == "untrend":
        data = untrend_data_df(data, columns)
    
    elif mode == "stationarize":
        data = stationnarize_data_df(data, columns)

    data = standardize_data_df(data, columns, with_std_mean=False)

    print(data)
    acp = PCA()
    
    acp_results = acp.fit_transform(data[columns])
    acp_results_df = pd.DataFrame(data=acp_results, 
                                    columns=[f"F{i+1}" for i in range(acp.n_components_)])
    n_cpt_kmeans = min(n_cpt_kmeans, acp.n_components_)
    k_means = KMeans(n_clusters=n_cluster)
    k_means.fit(acp_results_df[[f"F{i+1}" for i in range(n_cpt_kmeans)]])
    acp_results_df['label_cpt'] = k_means.labels_
    data['label_cpt'] = k_means.labels_

    explained_variance_ratio = acp.explained_variance_ratio_
    cumul_explained_variance_ratio = np.cumsum(explained_variance_ratio)
    
    fig = plotly.subplots.make_subplots(rows=2, cols=3,
                                        subplot_titles=(
                                            f"Données d'entrée ({input_plot})",
                                            "Pourcentage de variance expliquée avec somme cumulée", 
                                            "Individus représentés sur F1 et F2 "+\
                                                "de la nouvelle base", 
                                            "Contribution de chaque individus à l'inertie totale",
                                            "Cercle de corrélation (F1, F2)",
                                            "Matrice des corrélation variables / axes"
                                            ))
    fig.update_layout(
        title_text=f"ACP"+\
            f"{(f' - {name}' if not name is None else '')} - Colonnes étudiées : {columns}"+\
            f"- (mode : {mode})" )

    # 1 ###
    data_trace_df = data.copy()
    data_trace_df["color"] = data['label_cpt']

    input_data = px.scatter(data_trace_df, x="date", y=input_plot, 
                            color='color', 
                            text="date")["data"][0]
    input_data.mode = "lines+markers"
    input_data.line.width = 1
    input_data.marker.symbol = "circle"
    input_data.marker.size = 3
    fig.add_trace(input_data,
                  row=1, col=1)
    ####


    # 2 ###
    fig.add_trace(go.Bar(x=np.arange(1, len(explained_variance_ratio)+1),
                         y=explained_variance_ratio),
                  row=1, col=2)
    
    fig.add_trace(go.Scatter(x=np.arange(1, len(cumul_explained_variance_ratio)+1),
                             y=cumul_explained_variance_ratio,
                             mode="lines+markers"),
                  row=1, col=2)
    ####
    
    # 3 ###
    min_val = min(np.min(acp_results_df[acp_results_df.columns[0]]), 
                        np.min(acp_results_df[acp_results_df.columns[1]]))
    max_val = max(np.max(acp_results_df[acp_results_df.columns[0]]), 
                        np.max(acp_results_df[acp_results_df.columns[1]]))
    
    acp_results_trace_df = acp_results_df.copy()
    acp_results_trace_df["color"] = data['label_cpt']
    acp_results_trace_df["date"] = data['label_cpt']
    
    acp_results_trace = px.scatter(acp_results_trace_df,
                       x = acp_results_df.columns[0],
                       y = acp_results_df.columns[1],
                       color = 'color',
                       text="date")["data"][0]
    acp_results_trace.mode = 'markers'
    acp_results_trace.marker.symbol = "circle"
    acp_results_trace.marker.size = 3

    fig.add_trace(acp_results_trace,
                  row=1, col=3)
    fig.update_layout(xaxis3 = dict(range=[min_val, max_val]))
    fig.update_layout(yaxis3 = dict(range=[min_val, max_val]))
    ####
    
    total_inertie = (data[columns]**2).sum(axis=1)
    
    # 4 ###
    # total_inertie_trace_df = pd.DataFrame()
    # total_inertie_trace_df["inertie"] = total_inertie
    # total_inertie_trace_df["color"] = data['label_cpt']
    # total_inertie_trace_df["date"] = data_trace_df["date"]
    # trace_intertie = px.scatter(total_inertie_trace_df, 
    #                             x="date", 
    #                             y="inertie", color="color")["data"][0]
    # trace_intertie.mode = 'lines+markers'
    # trace_intertie.line.width = 1
    # trace_intertie.marker.symbol = "circle"
    # trace_intertie.marker.size = 3
    # fig.add_trace(trace_intertie,
    #               row=2, col=1)
    ####
    
    # qual_repr_indi = (acp_results**2)
    # for i in range(acp_results.shape[1]):
    #     qual_repr_indi[:,i] = qual_repr_indi[:,i] / total_inertie
    # qual_repr_indi_df = pd.DataFrame(qual_repr_indi, columns=columns)
    # qual_repr_indi_df["date"] = data_trace_df["date"]
    
    # contrib_axes = (acp_results**2)
    # for i in range(acp_results.shape[1]):
    #     contrib_axes[:,i] = contrib_axes[:,i] / acp_results.shape[0]*acp.explained_variance_[i]
    # contrib_axes_df = pd.DataFrame(contrib_axes, columns=columns)
    # contrib_axes_df["date"] = data_trace_df["date"]
    
    # 5 ###
    corvar = np.zeros((acp.n_components_, acp.n_components_))
    for k in range(acp.n_components_):
        # variable en ligne et facteurs en colonnes
        corvar[:,k] = acp.components_[k] * np.sqrt(acp.explained_variance_)[k]
    for var, col in zip(corvar, columns):
        fig.add_trace(go.Scatter(x=[0,var[0]], y=[0,var[1]], mode="lines+markers", text=col),
                    row=2, col=2)
    fig.add_shape(dict(type="circle", x0=-1, x1=1, y0=-1, y1=1, line_color="purple"), 
                  row=2, 
                  col=2)
    fig.layout.yaxis5.domain = (0.0, 0.5)
    fig.layout.xaxis5.domain = (0.38, 0.62)
    fig.layout.annotations[4].y = 0.5
    fig.layout.shapes[0].line.color = 'black'
    
    # 6 ###
    corvar_df = pd.DataFrame(corvar, 
                             index=columns,
                             columns=[f"F{i+1}" for i in range(acp.n_components_)])
    fig.add_trace(go.Heatmap(z=corvar, 
                             y=columns, 
                             x=[f"F{i+1}" for i in range(acp.n_components_)],
                             text=corvar,
                             visible=True,
                             colorscale="spectral"),
                  row=2, col=3)
    fig.layout.yaxis6.domain = (0.0, 0.6)
    fig.layout.annotations[5].y = -0.08
    # fig.data[10].colorbar = {'x':1, 'y' : 0.188, "len": 0.4}
    # fig.data[10].text = list(np.array(np.round(corvar, 2), dtype=str))
    ###
    
    fig.update(layout_showlegend=False)
    fig.update_coloraxes(showscale=False)
    if plot:
        fig.show()
    
    result = {}
    result["data"] = data
    result["kmeans"] = k_means
    result["fig"] = fig
    result["acp"] = acp
    result["coord"] = acp_results_df
    # result["total_inertie"] = total_inertie_trace_df[["inertie","date"]]
    # result["qual_repr"] = qual_repr_indi_df
    # result["contrib_axe"] = contrib_axes_df
    result["corvar"] = corvar_df
    return result



def populate_indicators(data : pd.DataFrame) -> pd.DataFrame:
    dataframe = data.copy()
    
    # dataframe['adx'] = ta.ADX(dataframe)
    # dataframe['rsi'] = ta.RSI(dataframe)
    # stoch_fast = ta.STOCHF(dataframe)
    # dataframe['fastd'] = stoch_fast['fastd']
    # dataframe['fastk'] = stoch_fast['fastk']
    # macd = ta.MACD(dataframe)
    # dataframe['macd'] = macd['macd']
    # dataframe['macdsignal'] = macd['macdsignal']
    # dataframe['macdhist'] = macd['macdhist']
    # dataframe['mfi'] = ta.MFI(dataframe)
    # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
    # dataframe['bb_lowerband'] = bollinger['lower']
    # dataframe['bb_middleband'] = bollinger['mid']
    # dataframe['bb_upperband'] = bollinger['upper']
    # dataframe["bb_percent"] = (
    #     (dataframe["close"] - dataframe["bb_lowerband"]) /
    #     (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
    # )
    # dataframe["bb_width"] = (
    #     (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
    # )
    # dataframe['sar'] = ta.SAR(dataframe)
    # dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
    # hilbert = ta.HT_SINE(dataframe)
    # dataframe['htsine'] = hilbert['sine']
    # dataframe['htleadsine'] = hilbert['leadsine']
    
    # dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
    # dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
    
    # ichimoku, ichimoku_forward = dataframe.ta.ichimoku(lookahead=False)
    # dataframe["ich_tenkan"] = ichimoku["ITS_9"]
    # dataframe["ich_kijun"] = ichimoku["IKS_26"]
    # dataframe["ich_spanA"] = ichimoku["ISA_9"]
    # dataframe["ich_spanB"] = ichimoku["ISB_26"]
    # dataframe["ich_chikou"] = dataframe["close"].shift(26)

    # hilbert = ta.HT_SINE(dataframe)
    # dataframe['htsine'] = hilbert['sine']
    # dataframe['htleadsine'] = hilbert['leadsine']
    
    # dataframe["ich_lead_spanA"] = pd.concat([ichimoku["ISA_9"], 
    #                                          ichimoku_forward["ISA_9"]]).shift(-26)
    # dataframe["ich_lead_spanB"] = pd.concat([ichimoku["ISB_26"], 
    #                                          ichimoku_forward["ISB_26"]]).shift(-26)
    
    # columns = sorted(list(set(data.columns) ^ set(["date"])))
    dataframe = add_past_shifted_columns(dataframe, list_columns = ["close"], n_shift=10)
        # dataframe[f"ich_tenkan{i}"] = dataframe["ich_tenkan"].shift(i)
        # dataframe[f"ich_kijun{i}"] = dataframe["ich_kijun"].shift(i)
        # dataframe[f"ich_spanA{i}"] = dataframe["ich_spanA"].shift(i)
        # dataframe[f"ich_spanB{i}"] = dataframe["ich_spanB"].shift(i)
        # dataframe[f"ich_chikou{i}"] = dataframe["ich_chikou"].shift(i)
        # dataframe[f"ich_lead_spanA{i}"] = dataframe["ich_lead_spanA"].shift(i)
        # dataframe[f"ich_lead_spanB{i}"] = dataframe["ich_lead_spanB"].shift(i)
        # dataframe[f'rsi{i}'] = dataframe['rsi'].shift(i)

    
    
    dataframe.dropna(inplace=True)
    return dataframe

def main():
    parser = argparse.ArgumentParser()
    
    ### A propos des données
    parser.add_argument("--pair_name", type=str, help="exemple : BTC/BUSD")
    parser.add_argument("-t", "--timeframe", type=str, help="exemple : 1m ")
    parser.add_argument("--days", type=int, default="30")
    parser.add_argument("--path", type=str, help="chemin vers les données", 
                        default="./user_data/data/binance")
    
    ### A propos de l'ACP et de l'analyse
    parser.add_argument("--col", type=str, default="all",
                        nargs='+', help="colonnes qui seront prise en compte ")
    parser.add_argument("--notcol", type=str, default="date",
                        nargs='+', help="colonnes qui ne seront pas prise en compte (if col == 'all') ")
    parser.add_argument("--input_ploted", type=str, 
                        help="la colonne qui sera affiché en tant qu'entré")
    parser.add_argument("--n_cluster", type=int, help="pour les kmeans")
    parser.add_argument("--n_cpt_kmeans", type=int, 
                        help="le nombre de facteur que les kmeans prendront en compte")
    parser.add_argument("--mode", type=str,
                        help="stationarize, untrend, default", default="stationarize")
    args = parser.parse_args()
    
    print(os.path.abspath(os.path.curdir))
    path = Path(args.path)
    if os.path.isdir(path):
        try:
            command = "freqtrade download-data --exchange binance"+ \
                f" --pairs '{args.pair_name}' --days '{args.days}' --timeframe '{args.timeframe}'"
            print(command)
            os.system(command)
        except:
            raise Exception("Le téléchargement des données a échoué")

        pair_history = load_pair_history(args.pair_name, args.timeframe, path)
    else:
        raise Exception(f"Le chemin est inconnu")

    data = populate_indicators(pair_history)
    
    if args.col == ["all"]:
        columns = sorted(list(set(data.columns) ^ set(args.notcol)))
    else:
        columns = args.col
        
    result_analyse = analyse_acp(data, columns,
                name = f"{args.pair_name}_{args.timeframe}", 
                mode = args.mode, 
                input_plot=args.input_ploted,
                n_cluster=args.n_cluster,
                n_cpt_kmeans=args.n_cpt_kmeans,
                plot=True)

    for name in result_analyse:
        print(result_analyse[name])
        
if __name__ == "__main__":
    main()