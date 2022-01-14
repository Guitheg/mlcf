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

from sklearn.linear_model import LinearRegression

import plotly
import plotly.express as px
import plotly.graph_objects as go

from utils import data_trend_one_axis, standardize_data_df, stationnarize_data_df

def populate_indicators(data : pd.DataFrame) -> pd.DataFrame:
    dataframe = data.copy()
    
    dataframe.dropna(inplace=True)
    return dataframe

def build_graphs(path, pair_name, days, timeframe, col, notcol):
    path = Path(path)
    if os.path.isdir(path):
        try:
            command = "freqtrade download-data --exchange binance"+ \
                f" --pairs '{pair_name}' --days '{days}' --timeframe '{timeframe}'"
            print(command)
            os.system(command)
        except:
            raise Exception("Le téléchargement des données a échoué")

        pair_history = load_pair_history(pair_name, timeframe, path)
    else:
        raise Exception(f"Le chemin est inconnu")

    data = populate_indicators(pair_history)
    
    if col == ["all"]:
        columns = sorted(list(set(data.columns) ^ set(notcol)))
    else:
        columns = col

    data["mid"] = data[['high', 'low']].mean(axis=1)
    columns.append("mid")
    
    data_dt = stationnarize_data_df(data, columns)
    
    data = standardize_data_df(data, columns)
    
    data_dt = standardize_data_df(data_dt, columns)
    
    mean_mid = data_dt["mid"].mean()

    # data_dt["volatilite"] = [((data_dt["mid"].tail(i) - mean_mid)**2).sum()/i for i in range(len(data_dt))]
    data_dt["abs"] = data_dt["mid"].abs()
    data_dt["volatilite"] = np.array([data_dt["abs"].iloc[i-100:i].var() for i in range(len(data_dt["abs"]))])
    # data_vol = pd.DataFrame(data_dt["volatilite"], columns=["volatilite"])
    corr_win_size = 500
    data_corr_ret_vol = pd.DataFrame(np.array([data_dt["mid"].corr(data_dt["volatilite"].shift(-i)) for i in range(-corr_win_size, corr_win_size)]), columns=["corr_ret_vol"])
    data_vol = pd.DataFrame(np.array([data_dt["abs"].tail(i).var() for i in range(5000)]), columns=["volatilite"])

    data_corr = pd.DataFrame(np.array([data_dt["abs"].autocorr(t) for t in range(500)]), columns=["autocorr"])

    
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
    
    trace_data = go.Scatter(x=data["date"], y=data["mid"], name=f"({pair_name}) C(t)")
    figures.append((trace_data, 
                    f"Cours", 
                   "Temps - t", 
                   "Prix - C(t)"))
    
    trace_data_dt = go.Scatter(x=data_dt["date"], y=data_dt["mid"], name=f"({pair_name}) dC(t)/dt")
    figures.append((trace_data_dt, 
                    f"Variation du cours", 
                    "Temps - t", 
                    "Variation Prix - dC(t)/dt"))
    
    trace_data_dt_abs = go.Scatter(x=data_dt["date"], y=data_dt["abs"], name = f"({pair_name}) abs(dC(t)/dt)")
    figures.append((trace_data_dt_abs, 
                    f"Variation absolu du cours", 
                    "Temps - t", 
                    "Variation Prix - abs(dC(t)/dt)"))
    
    count, values = np.histogram(data_dt["mid"], bins = max(len(data_dt)//cumul_value, 1)) 
    trace_histogram_data_dt = go.Scatter(x=values[:-1], y=count,  name=f"({pair_name}) Nombre par dC(t)/dt")
    figures.append((trace_histogram_data_dt, 
                    f"Histogramme des variations", 
                   "dC(t)/dt", 
                   "Nombre"))
    
    count, values = np.histogram(data_dt["abs"], bins = max(len(data_dt)//cumul_value, 1)) 
    trace_histogram_data_dt_abs = go.Scatter(x=values[:-1], y=count, name=f"({pair_name}) Nombre par abs(dC(t)/dt)") 
    figures.append((trace_histogram_data_dt_abs, 
                    f"Histogramme des variations absolues", 
                   "abs(dC(t)/dt)", 
                   "Nombre"))
    
    
    trace_cdf = go.Scatter(x=data_dt_stats["log_var_abs"], y=data_dt_stats["log_cdf"], name = f"({pair_name}) Log(Prob[x<=abs(dC(t)/dt)])")
    figures.append((trace_cdf, 
                    f"Probabilité cumulé (log)", 
                   "Variation - Log(abs(dC(t)/dt))",
                   "Probabilité cumulé - Log(Prob[x<=abs(dC(t)/dt)])" ))
    
    trace_vol = go.Scatter(x=data_dt["date"], y=data_dt["volatilite"], name = f"({pair_name}) Var(dC(t)/dt)")
    figures.append((trace_vol, 
                    f"Evolution de la volatilité",
                    "Temps - t",
                    "Volatilité - Var(dC(t)/dt)"
                   ))
    
    trace_vol = go.Scatter(x=data_vol.index, y=data_vol["volatilite"], name = f"({pair_name}) Var(dC(t)/dt)")
    figures.append((trace_vol, 
                    f"Volatilité calculé sur une fenêtre de T",
                    "Temps - T",
                    "Volatilité - Var(dC(t)/dt)"
                   ))
    
    
    trace_autocorr = go.Scatter(x=np.log10(data_corr.index), y=np.log10(data_corr["autocorr"]), name = f"({pair_name}) log10(Corr[r²_t, r²_t+T])")
    figures.append((trace_autocorr, 
                    f"Log des Autocorrelations de la variation absolu à T",
                    "Temps - log10(T)",
                    "Auto-corrélation à T - log10(Corr[r²_t, r²_t+T])"
                   ))
    
    trace_autocorr = go.Scatter(x=data_corr.index, y=data_corr["autocorr"], name = f"({pair_name}) Corr[r²_t, r²_t+T]")
    figures.append((trace_autocorr, 
                    f"Autocorrelations de la variation absolu à T",
                    "Temps - T",
                    "Auto-corrélation à T - Corr[r²_t, r²_t+T]"
                   ))
    
    trace_corr_ret_vol = go.Scatter(x=data_corr_ret_vol.index-corr_win_size, y=data_corr_ret_vol["corr_ret_vol"], name = f"({pair_name}) Corr[R, V.shift(-T)]")
    figures.append((trace_corr_ret_vol, 
                    f"Corrélation entre variation et volatilité",
                    "Temps - T",
                    "Corrélation - Corr[R, V.shift(-T)] "
                   ))
    
    
    return figures

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

    graphes = []
    for pair in args.pair_names:
        graph = build_graphs(args.path, pair, args.days, args.timeframe, args.col, args.notcol)
        graphes.append(graph)
      
    height_by_row = 500
    c = 3
    r = (len(graphes[0]) // c)+1  
    window = plotly.subplots.make_subplots(rows=r, cols=c, 
                                           subplot_titles=tuple(
                                            [ f"{title}" for (_, title,_,_) in graphes[0]]
                                            ))
    window.update_layout(title_text=f"Etude sur {args.pair_names}")
    window.update_layout(height=height_by_row*r)
    for figures in graphes : 
        for i, (trace, _,_,_) in enumerate(figures):
            # for trace_i in range(len(trace.data)):
                # if "line" in trace.data[trace_i]:
                #     trace.data[trace_i].line.color = px.colors.qualitative.Alphabet[num_pair]
                # elif "marker" in trace.data[trace_i]:
                #     trace.data[trace_i].marker.color = px.colors.qualitative.Alphabet[num_pair]
            window.add_trace(trace, row=(i//c)+1, col=(i%c)+1)
    
    ## droite indicatrice :
    x = np.linspace(1, 10, 100)
    window.add_trace(go.Scatter(x = np.log10(x),
                                   y = np.log10(1/x**3), name="1/x^3"
                                   ), row = 2, col = 3)
    
    x = np.linspace(0, 500, 100)
    window.add_trace(go.Scatter(x = np.log10(x),
                                y = np.log10((x**-0.4)), name="T^-0.4"
                                ), row = 3, col = 3)
    
    x = np.linspace(0, 500, 100)
    window.add_trace(go.Scatter(x = x,
                                y = (x**-0.4), name="T^-0.4"
                                ), row = 4, col = 1)
            
    if len(graphes[0]) > 1:
        window.layout.xaxis["title"] = graphes[0][0][2]
        window.layout.yaxis["title"] = graphes[0][0][3]
        for num_pair, figures in enumerate(graphes):
            for trace in range(1, len(figures)):
                window.layout[f"xaxis{trace+1}"]["title"] = figures[trace][2]
                window.layout[f"yaxis{trace+1}"]["title"] = figures[trace][3]
    window.update(layout_showlegend=True)
    window.show()

if __name__ == "__main__":
    main()