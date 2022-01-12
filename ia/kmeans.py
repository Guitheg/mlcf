# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.data.history.history_utils import load_pair_history

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

from pandas_ta.overlap.ichimoku import ichimoku
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import plotly
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA

import argparse
from pca import analyse_acp

from utils import data_trend_one_axis, standardize_data_df

def populate_indicators(data : pd.DataFrame) -> pd.DataFrame:
    dataframe = data.copy()
    
    dataframe.dropna(inplace=True)
    return dataframe

def extremum_local(data : pd.DataFrame, window_size : int):
    pass


def post_treatment_peaks(extremum, i = 0):
    if i == 0:
        sys.setrecursionlimit(max(1000, len(extremum)))
        
    length = len(extremum.index)-1

    if i >= length:
        return extremum
    
    idx = extremum.index[i]
    idx_1 = extremum.index[i+1]
    if extremum.loc[idx]["label"] == extremum.loc[idx_1]["label"]:
        if extremum.loc[idx]["label"] == "min":
            if extremum.loc[[idx, idx_1]]["extremum"].argmin() == 0:
                extremum.drop(idx_1, inplace=True)
                return post_treatment_peaks(extremum, i)
            else:
                extremum.drop(idx, inplace=True)
                return post_treatment_peaks(extremum, i+1)
        if extremum.loc[idx]["label"] == "max":
            if extremum.loc[[idx, idx_1]]["extremum"].argmax() == 0:
                extremum.drop(idx_1, inplace=True)
                return post_treatment_peaks(extremum, i)
            else:
                extremum.drop(idx, inplace=True)
                return post_treatment_peaks(extremum, i+1)         
        else:
            return post_treatment_peaks(extremum, i+1)
        
    if ((extremum.loc[idx]["label"] == "max" and extremum.loc[idx_1]["label"] == "min" and
        extremum.loc[idx]["extremum"] <= extremum.loc[idx_1]["extremum"]) or 
        (extremum.loc[idx]["label"] == "min" and extremum.loc[idx_1]["label"] == "max" and
        extremum.loc[idx]["extremum"] >= extremum.loc[idx_1]["extremum"])):
        
        extremum.drop(idx, inplace=True)
        extremum.drop(idx_1, inplace=True)
        return post_treatment_peaks(extremum, i) 
    
    # if ((extremum.loc[idx]["label"] != extremum.loc[idx_1]["label"]) and
    #     abs(extremum.loc[idx]["extremum"] - extremum.loc[idx_1]["extremum"]) > tol):
        
    #     extremum.drop(idx, inplace=True)
    #     extremum.drop(idx_1, inplace=True)
    #     return post_treatment_peaks(extremum, i) 
    
    else:
        return post_treatment_peaks(extremum, i+1)

def get_list_pattern(data : pd.DataFrame, size_pattern : int) -> pd.DataFrame:
    dataframe = data.copy()
    list_pattern = np.array([np.asarray(dataframe.extremum.shift(-i).head(size_pattern)) for i in range(len(dataframe)-size_pattern)])
    list_pattern = StandardScaler().fit_transform(list_pattern.T).T
    return pd.DataFrame(list_pattern, columns=[f"P{i}" for i in range(size_pattern)])

def main():
    parser = argparse.ArgumentParser()
    
    ### A propos des données
    parser.add_argument("--pair_name", type=str, help="exemple : BTC/BUSD")
    parser.add_argument("-t", "--timeframe", type=str, help="exemple : 1m ")
    parser.add_argument("--days", type=int, default="30")
    parser.add_argument("--path", type=str, help="chemin vers les données", 
                        default="./user_data/data/binance")
    parser.add_argument("--col", type=str, default=["all"],
                        nargs='+', help="colonnes qui seront prise en compte ")
    parser.add_argument("--notcol", type=str, default=["date"],
                        nargs='+', help="colonnes qui ne seront pas prise en compte (if col == 'all') ")
    parser.add_argument("--prominence", type=float, default=0.15, help="affinage des dents de scies pour les patterns")
    parser.add_argument("--n_clusters", type=int, default=3, help = "nombre cluser kmeans")
    parser.add_argument("--size_pattern", type=int, help="taille d'un pattern (en points)")
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

    data = standardize_data_df(data, columns)

    from scipy.signal import find_peaks
    
    data["mean"] = data[['high', 'low']].mean(axis=1)
    

    
    peaks_max = list(find_peaks(data["mean"], prominence=args.prominence))[0]
    peaks_min = list(find_peaks(data["mean"]*-1, prominence=args.prominence))[0]

    extremum_max = pd.DataFrame(np.array(data.iloc[peaks_max]["mean"]), columns=["extremum"], index=peaks_max)
    extremum_max["date"] = data["date"].iloc[peaks_max]
    extremum_max["label"] = "max"
    extremum_min = pd.DataFrame(np.array(data.iloc[peaks_min]["mean"]), columns=["extremum"], index=peaks_min)
    extremum_min["date"] = data["date"].iloc[peaks_min]
    extremum_min["label"] = "min"
    extremum = pd.concat([extremum_max, extremum_min])
    extremum.sort_values("date", inplace=True)
    
    # extremum = post_treatment_peaks(extremum)

    ext = px.scatter(extremum, x='date', y="extremum", color="label")
    line = px.scatter(extremum, x='date', y="extremum")
    line.data[0].mode="lines"
    line.data[0].line.color = "black"
    ext.add_trace(line.data[0])
    ext.add_trace(go.Candlestick(x = data['date'],
                open = data['open'],
                high = data['high'],
                low = data['low'],
                close = data['close']))
    # ext.show()
    
    list_pattern = get_list_pattern(extremum, args.size_pattern)
    
    # kmeans = KMeans(n_clusters=args.n_clusters)
    # result_kmeans = kmeans.fit(list_pattern)
    

    
    # acp = PCA()
    # print(list_pattern.iloc[:, 0:args.size_pattern])
    # acp_results = acp.fit_transform(list_pattern.iloc[:, 0:args.size_pattern])
    # acp_results_df = pd.DataFrame(data=acp_results, 
    #                                 columns=[f"F{i+1}" for i in range(acp.n_components_)])
    # print(acp_results_df)
    # acp_results_df["color"] = list_pattern["label"]

    # acp_results_trace = px.scatter(acp_results_df,
    #                    x = acp_results_df.columns[0],
    #                    y = acp_results_df.columns[1],
    #                    color = 'color')
    # acp_results_trace["data"][0].mode = 'markers'
    # acp_results_trace["data"][0].marker.symbol = "circle"

    # acp_results_trace.show()
    
    result = analyse_acp(list_pattern, [f"P{i}" for i in range(args.size_pattern)], n_cluster=args.n_clusters)
    list_pattern["label"] = result["kmeans"].labels_
    
    height_by_row = 300
    c = 6
    r = (len(list_pattern) // c)
    
    fig = plotly.subplots.make_subplots(rows=r, cols=c, 
                                        subplot_titles=tuple(
                                            [ f"{i} label : {result['kmeans'].labels_[i]}" for i in range(r*c)]
                                            )
                                        )
    fig.update_layout(title_text=f"Liste des patterns (prominence : {args.prominence})")
    fig.update_layout(height=height_by_row*r)
    for i in range(r):
        for j in range(c):
            series = list_pattern.iloc[(i*c)+j][0:args.size_pattern]
            color = int(list_pattern.iloc[(i*c)+j]["label"])

            line = px.scatter(x=np.arange(len(series)), y=series)
            line.data[0].mode = "lines+markers"
            line.data[0].line.color = px.colors.qualitative.Alphabet[color]
            fig.add_trace(line.data[0],
                  row=i+1, col=j+1)
    fig.show()
    
if __name__ == "__main__":
    main()
    
# Modélisation du problème :
# Agrégation de stratégie / agent qui simule le cours
# agent modélisant une catégorie de comportement (qui devront ne pas converger vers la meme strat)
# Le système évolue au cours du temps
# capter l'inertie du système pour prédire comment le système va évoluer