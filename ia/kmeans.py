# -*- coding: utf-8 -*-

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

import argparse

from utils import data_trend_one_axis, standardize_data_df

def populate_indicators(data : pd.DataFrame) -> pd.DataFrame:
    dataframe = data.copy()
    
    dataframe.dropna(inplace=True)
    return dataframe

def extremum_local(data : pd.DataFrame, window_size : int):
    pass


def post_treatment_peaks(extremum, i = 0):
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
        if (extremum.loc[idx]["label"] == "max" and extremum.loc[idx_1]["label"] == "min" and
            extremum.loc[idx]["extremum"] <= extremum.loc[idx_1]["extremum"]):
            
            extremum.drop(idx, inplace=True)
            extremum.drop(idx_1, inplace=True)
            return post_treatment_peaks(extremum, i+2) 
        
        if (extremum.loc[idx]["label"] == "min" and extremum.loc[idx_1]["label"] == "max" and
            extremum.loc[idx]["extremum"] >= extremum.loc[idx_1]["extremum"]):
            
            extremum.drop(idx, inplace=True)
            extremum.drop(idx_1, inplace=True)
            return post_treatment_peaks(extremum, i+2) 
        
        else:
            return post_treatment_peaks(extremum, i+1)

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
    
    args = parser.parse_args()
    
    print(os.path.abspath(os.path.curdir))
    path = Path(args.path)
    if os.path.isdir(path):
        # try:
        #     command = "freqtrade download-data --exchange binance"+ \
        #         f" --pairs '{args.pair_name}' --days '{args.days}' --timeframe '{args.timeframe}'"
        #     print(command)
        #     os.system(command)
        # except:
        #     raise Exception("Le téléchargement des données a échoué")

        pair_history = load_pair_history(args.pair_name, args.timeframe, path)
    else:
        raise Exception(f"Le chemin est inconnu")

    data = populate_indicators(pair_history)
    
    if args.col == ["all"]:
        columns = sorted(list(set(data.columns) ^ set(args.notcol)))
    else:
        columns = args.col

    data = standardize_data_df(data, columns)
        
    # fig = go.Figure(data=[go.Candlestick(x = data['date'],
    #             open = data['open'],
    #             high = data['high'],
    #             low = data['low'],
    #             close = data['close'])])
    # fig.show()
    
    from scipy.signal import find_peaks
    
    data["mean"] = data[['high', 'low']].mean(axis=1)
    
    peaks_max = list(find_peaks(data["mean"], distance=15))[0]
    peaks_min = list(find_peaks(data["mean"]*-1, distance=15))[0]
    # print(peaks)
    
    # peaks_df = pd.DataFrame(peaks, columns=["extremum"])
    extremum_max = pd.DataFrame(np.array(data.iloc[peaks_max]["mean"]), columns=["extremum"], index=peaks_max)
    extremum_max["date"] = data["date"].iloc[peaks_max]
    extremum_max["label"] = "max"
    extremum_min = pd.DataFrame(np.array(data.iloc[peaks_min]["mean"]), columns=["extremum"], index=peaks_min)
    extremum_min["date"] = data["date"].iloc[peaks_min]
    extremum_min["label"] = "min"
    extremum = pd.concat([extremum_max, extremum_min])
    extremum.sort_values("date", inplace=True)
    ext = px.scatter(extremum, x='date', y="extremum", color="label")
    
    fig = px.scatter(data, x='date', y="mean")
    fig.data[0].mode="lines"
    fig.data[0].line.width = 0.5
    fig.data[0].line.color = "black"
    ext.add_trace(fig.data[0])
    ext.show()
    
    extremum = post_treatment_peaks(extremum)
    
    ext = px.scatter(extremum, x='date', y="extremum", color="label")
    
    fig = px.scatter(data, x='date', y="mean")
    fig.data[0].mode="lines"
    fig.data[0].line.width = 0.5
    fig.data[0].line.color = "black"
    ext.add_trace(fig.data[0])
    ext.show()
            
    # print(
    #     extremum[
    #         (extremum["label"] == extremum["label"].shift(1)) & 
    #         (extremum["label"] == "min")
    #         ].index.values
    #     )
    # print(extremum)


    
    # ext = px.scatter(extremum, x='date', y="extremum", color="label")
    
    # fig = px.scatter(data, x='date', y="mean")
    # fig.data[0].mode="lines"
    # fig.data[0].line.width = 0.5
    # fig.data[0].line.color = "black"
    # ext.add_trace(fig.data[0])
    # ext.show()
    
    
    
if __name__ == "__main__":
    main()