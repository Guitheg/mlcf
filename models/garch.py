import os
import argparse
from pathlib import Path
import arch

from freqtrade.data.history.history_utils import load_pair_history
from pandas_ta.overlap.ichimoku import ichimoku
from sklearn.preprocessing import StandardScaler
from sqlalchemy import column
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts


from matplotlib import pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go

from arch import arch_model
from arch.__future__ import reindexing

def split_pandas(main_data, proportion):
    times = sorted(main_data.index.values)
    second_part = sorted(main_data.index.values)[-int(proportion*len(times))]
    second_data = main_data[(main_data.index >= second_part)]
    first_data = main_data[(main_data.index < second_part)]
    return first_data, second_data

def populate_indicators(data : pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    dataframe = data.copy()
    dataframe["p"] = ta.SMA(dataframe, timeperiod=2)
    # dataframe["r"] = np.log(dataframe.p / dataframe.p.shift(1))
    dataframe["r"] = dataframe.p.pct_change()*100
    dataframe["o"] = dataframe.r.rolling(6).std()*((6*365)**0.5)
    dataframe.dropna(inplace=True)
    return dataframe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_name", type=str, help="exemple : BTC/BUSD", default="BTC/BUSD")
    parser.add_argument("--key", type=str, help="is set to 'date by default", default=["date"], nargs='+')
    parser.add_argument("-t", "--timeframe", type=str, help="exemple : 1m ", default="4h")
    parser.add_argument("--days", type=int, default="30")
    parser.add_argument("--path", type=str, help="chemin vers les données", 
                        default="./user_data/data/binance")
    parser.add_argument("--col", type=str, default=["all"],
                        nargs='+', help="colonnes qui seront prise en compte ")
    parser.add_argument("--notcol", type=str, default=[],
                        nargs='+', help="colonnes qui ne seront pas prise en compte (if col == 'all') ")
    args = parser.parse_args()

    path = Path(args.path)
    pair = args.pair_name
    col = args.col
    t = args.timeframe
    
    pair_history = load_pair_history(pair, t, path)
    # pair_history.set_index(args.key, inplace=True)
    # pair_history = pair_history.loc['2020-12-22':]
    data = populate_indicators(pair_history, t=t)
    
    if col == ["all"]:
        columns = sorted(list(set(data.columns) ^ set(args.notcol) ^ set(args.key)))
    else:
        columns = col
    
    # print(columns)
    train_data, valid_data = split_pandas(data, 0.2)
    
    # train_data[columns] = StandardScaler(with_mean=False).fit_transform(train_data[columns])
    # valid_data[columns] = StandardScaler(with_mean=False).fit_transform(valid_data[columns])
    
    garch_model_train = arch_model(train_data.r*1000, mean="Zero", vol="GARCH", p=2, o=0, q=1, dist='Normal').fit()
    
    prediction = garch_model_train.forecast(horizon=1000, reindex=False)
    print(prediction.variance.tail())
    predicted_variance = pd.DataFrame(index=np.arange(prediction.variance.index[0], 
                                                      prediction.variance.index[0]+
                                                      len(prediction.variance.T)))
    
    predicted_variance["pred_var"] = (prediction.variance.T.values**0.5)*((6*365)**0.5)/1000
    
    for i in garch_model_train.conditional_volatility.index:
        predicted_variance.loc[i,"obs_var"] = ((garch_model_train.conditional_volatility[i])*((6*365)**0.5)/1000)

    ### DESSINS COURBE
    figures = []
    
    trace_data = go.Scatter(x=data.index, y=data.p, name=f"Prix")
    figures.append((trace_data, 
                    f"Prix", 
                   "t", 
                   "p(t)"))
    
    trace_data = go.Scatter(x=data.index, y=data.r, name=f"Retour")
    figures.append((trace_data, 
                    f"Retour", 
                   "t", 
                   "r(t)"))
    
    trace_data = go.Scatter(x=data.index, y=data.o, name=f"Volatilité - fenetre d'une journée")
    figures.append((trace_data, 
                    f"Volatilité - fenetre d'une journée", 
                   "t", 
                   "std(r(t))[6]"))
    
    pred_plot = go.Scatter(x=predicted_variance.index, y=predicted_variance["pred_var"], name=f"PREDICTION - var(r(t))")
    obs_plot = go.Scatter(x=predicted_variance.index, y=predicted_variance["obs_var"], name=f"PREDICTION - var(r(t))")
    
    ### PLOT ###
    height_by_row = 500
    c = 3
    r = (len(figures) // c)+1
    
    window = plotly.subplots.make_subplots(rows=r, cols=c, 
                                           subplot_titles=tuple(
                                            [ f"{title}" for (_, title,_,_) in figures]
                                            ))
    window.update_layout(height=height_by_row*r)
    for i, (trace, _,_,_) in enumerate(figures):
        window.add_trace(trace, row=(i//c)+1, col=(i%c)+1)
    
    window.add_trace(pred_plot,row=1, col=3)
    window.add_trace(obs_plot,row=1, col=3)
    
    # window.update_traces(showscale=False)
    if len(figures) > 1:    
        for trace in range(1, len(figures)):
            window.layout[f"xaxis{trace+1}"]["title"] = figures[trace][2]
            window.layout[f"yaxis{trace+1}"]["title"] = figures[trace][3]
    if len(figures) > 0:
        window.layout.xaxis["title"] = figures[0][2]
        window.layout.yaxis["title"] = figures[0][3]
        window.show()

if __name__ == "__main__":
    main()