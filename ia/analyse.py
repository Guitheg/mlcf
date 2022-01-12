# -*- coding: utf-8 -*-

import os
from pathlib import Path
import argparse

import pandas as pd
import numpy as np

from freqtrade.data.history.history_utils import load_pair_history
from sklearn.preprocessing import StandardScaler

from pandas_ta.overlap.ichimoku import ichimoku
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import plotly
import plotly.express as px
import plotly.graph_objects as go

from utils import data_trend_one_axis, standardize_data_df, stationnarize_data_df

def populate_indicators(data : pd.DataFrame) -> pd.DataFrame:
    dataframe = data.copy()
    
    dataframe.dropna(inplace=True)
    return dataframe

def main():
    parser = argparse.ArgumentParser()
    
    ### A propos des données
    parser.add_argument("--pair_names", type=str, help="exemple : BTC/BUSD", nargs='+')
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
    pair_histories = {}
    if os.path.isdir(path):
        for pair in args.pair_names:
            try:
                command = "freqtrade download-data --exchange binance"+ \
                    f" --pairs '{pair}' --days '{args.days}' --timeframe '{args.timeframe}'"
                print(command)
                os.system(command)
            except:
                raise Exception("Le téléchargement des données a échoué")

            pair_histories[pair] = load_pair_history(pair, args.timeframe, path)
            pair_histories[pair] = populate_indicators(pair_histories[pair])
            if args.col == ["all"]:
                columns = sorted(list(set(pair_histories[pair].columns) ^ set(args.notcol)))
            else:
                columns = args.col
    else:
        raise Exception(f"Le chemin est inconnu")


    for pair in pair_histories : 
        data = pair_histories[pair]
        data["mid"] = data[['high', 'low']].mean(axis=1)
        columns.append("mid")
        
        data_dt = stationnarize_data_df(data, columns)
        
        data = standardize_data_df(data, columns)
        
        data_dt = standardize_data_df(data_dt, columns)
        
        mean_mid = data_dt["mid"].mean()
        data_dt["volatilite"] = [np.sqrt(((data_dt["mid"].iloc[:i] - mean_mid)**2).sum()/len(data_dt)) for i in range(len(data_dt))]
        data_dt["abs"] = data_dt["mid"].abs()
        
        data_dt_stats = pd.DataFrame()
        cumul_value = 10
        count, values = np.histogram(data_dt["abs"], bins = max(len(data_dt)//cumul_value, 1)) 
        data_dt_stats["frequency"] = count
        data_dt_stats["var_abs"] = values[:-1]
        data_dt_stats["pdf"] = data_dt_stats["frequency"] / data_dt_stats["frequency"].sum()
        data_dt_stats['cdf'] = 1 - data_dt_stats["pdf"].cumsum() # prob(x>=)
        data_dt_stats["log_cdf"] = np.log10(data_dt_stats["cdf"])
        data_dt_stats["log_var_abs"] = np.log10(data_dt_stats["var_abs"])
    
    figures = []
    
    trace_data = px.line(data, x="date", y="mid")
    figures.append((trace_data, 
                    f"Cours du {args.pair_name}", 
                   "Temps - t", 
                   "Prix - C(t)"))
    
    trace_data_dt = px.line(data_dt, x="date", y="mid")
    figures.append((trace_data_dt, 
                    f"Variation du cours {args.pair_name}", 
                    "Temps - t", 
                    "Variation Prix - dC(t)/dt"))
    
    trace_data_dt_abs = px.line(data_dt, x="date", y="abs")
    figures.append((trace_data_dt_abs, 
                    f"Variation absolu du cours {args.pair_name}", 
                    "Temps - t", 
                    "Variation Prix - abs(dC(t)/dt)"))
    
    trace_histogram_data_dt = px.histogram(data_dt, x="mid")
    figures.append((trace_histogram_data_dt, 
                    f"Histogramme des variations {args.pair_name}", 
                   "dC(t)/dt", 
                   "Nombre"))
    
    trace_histogram_data_dt_abs = px.histogram(data_dt, x="abs")
    figures.append((trace_histogram_data_dt_abs, 
                    f"Histogramme des variations absolues {args.pair_name}", 
                   "abs(dC(t)/dt)", 
                   "Nombre"))
    
    
    trace_cdf = px.line(data_dt_stats, x="log_var_abs", y="log_cdf")
    figures.append((trace_cdf, 
                    f"Probabilité cumulé (log) {args.pair_name}", 
                   "Variation - Log(abs(dC(t)/dt))",
                   "Probabilité cumulé - Log(Prob[x<=abs(dC(t)/dt)])" ))
    
    
    
    trace_vol = px.line(data_dt, x="date", y="volatilite")
    figures.append((trace_vol, 
                    f"Evolution de la volatilité {args.pair_name}",
                    "Temps - t",
                    "Volatilité - Var_{E[C]}(dC(t)/dt)"
                   ))
    height_by_row = 500
    c = 3
    r = (len(figures) // c)+1
    
    window = plotly.subplots.make_subplots(rows=r, cols=c, 
                                           subplot_titles=tuple(
                                            [ f"{title}" for (_, title,_,_) in figures]
                                            ))
    window.update_layout(height=height_by_row*r)
    for i, (trace, _,_,_) in enumerate(figures):
        for trace_i in range(len(trace.data)):
            window.add_trace(trace.data[trace_i], row=(i//c)+1, col=(i%c)+1)
            
    window.layout.xaxis["title"] = figures[0][2]
    window.layout.yaxis["title"] = figures[0][3]
    for trace in range(1, len(figures)):
        window.layout[f"xaxis{trace+1}"]["title"] = figures[trace][2]
        window.layout[f"yaxis{trace+1}"]["title"] = figures[trace][3]
    window.show()
    
if __name__ == "__main__":
    main()