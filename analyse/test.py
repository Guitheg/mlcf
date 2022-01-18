from dataclasses import dataclass
import os
import argparse
from pathlib import Path
from functools import reduce

from freqtrade.data.history.history_utils import load_pair_history
# from .ia.utils import standardize_data_df

from pandas_ta.overlap.ichimoku import ichimoku
from sklearn.preprocessing import StandardScaler
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import pandas as pd

import plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import statsmodels.tsa.stattools as ts
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot as plt

W_SIZE_VOL = 20


tf = {"1m" : 60}
tf["5m"] = 5 * tf["1m"]
tf["15m"] = 15 * tf["1m"]
tf["30m"] = 30 * tf["1m"]
tf["1h"] = 60 * tf["1m"]
tf["4h"] = 4 * tf["1h"]
tf["1d"] = 24 * tf["1h"]
tf["1w"] = 7 * tf["1d"]

def f_given_tf(t : str) -> float:
    return 1/tf[t]
    
def populate_indicators(data : pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    dataframe = data.copy()
    
    # Prix p
    # dataframe["p"] = data[['high', 'low']].mean(axis=1)
    dataframe["p"] = ta.SMA(dataframe, timeperiod=5)
    print(ts.adfuller(dataframe.p.dropna()))
    
    # Return r - Variation p(t)/dt
    dataframe["r"] = np.log(dataframe.p / dataframe.p.shift(1))
    
    print(ts.adfuller(dataframe.r.dropna()))
    
    cov_r = pd.Series()
    for i in dataframe.index:
        cov_r.loc[i] = dataframe.loc[:i].r.cov(dataframe.r.shift(1).loc[:i])
    dataframe["cov_r"] = cov_r
    
    dataframe["var_r"] = dataframe.r.expanding().var()
    
    # Volatility from volume
    dataframe["dv"] = np.log(dataframe.volume / dataframe.volume.shift(1))
    
    # Volatility
    u = dataframe.high - dataframe.open
    d = dataframe.low - dataframe.open
    c = dataframe.close - dataframe.open
    a = 0.12
    if "t" in kwargs:
        f = f_given_tf(kwargs["t"])
    else:
        f = 1/3600
    # dataframe["o"] 
    o = (0.511*((u - d)**2)) - (0.019*(c*(u+d) - 2*u*d)) - (0.383*(c**2))
    dataframe["o"] = (a * (dataframe.open - dataframe.close.shift(1))**2 / f) + ((1-a)*o/(1-f))
    dataframe[f"o[{W_SIZE_VOL}]"] = np.sqrt(dataframe.r.rolling(window=W_SIZE_VOL).var())
    # dataframe["o"] = dataframe.r.expanding().var()
    
    # Volatily derivative
    dataframe[f"do[{W_SIZE_VOL}]"] = dataframe[f"o[{W_SIZE_VOL}]"] / dataframe[f"o[{W_SIZE_VOL}]"].shift(1)
    dataframe["do"] = dataframe.o / dataframe.o.shift(1)
    # dataframe["log_do"] = np.log10(dataframe.do)
    
    # Next Volatily
    # dataframe[f"no[{W_SIZE_VOL}]"] = dataframe[f"o[{W_SIZE_VOL}]"].shift(-30)
    dataframe["no"] = dataframe.o.shift(-30)
    
    dataframe.dropna(inplace=True)
    return dataframe


def corr_do_no(data_s : pd.Series, 
              doT_range : tuple = (5, 200, 5), 
              noT_range : tuple = (1, 50, 1)) -> pd.DataFrame:
    """Calcul la moving variance v avec data avec une fenetre vT
    Calcul la moving variance à nvT dans le future de la moving variance v 
    
    (Note la variance d'une variation d'un cours est égale à la volatilité)

    Args:
        data_r ([type]): [description]
    """
    data = data_s.copy()
    correlation_do_no = pd.DataFrame()
    
    for noTi in range(noT_range[0], noT_range[1], noT_range[2]):
        for doTi in range(doT_range[0], doT_range[1], doT_range[2]):
            o = np.sqrt(data.rolling(window=doTi).var()).dropna()
            no = o.shift(-noTi).dropna()
            for_corr = pd.DataFrame()
            for_corr['do'] = o / o.shift(1)
            for_corr['no'] = no
            correlation_do_no.loc[noTi, doTi] = for_corr.corr().iloc[0,1]
    
    return correlation_do_no
    
def o_tau(data_s : pd.Series, n_tau : int = 2000) -> pd.DataFrame:
    data = data_s.copy()
    o_T = []
    for i in range(1, n_tau):
        o_T.append(np.sqrt(data.iloc[::i].var()))
        
    return pd.DataFrame(o_T, columns=["o[T](t)"])    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_names", type=str, help="exemple : BTC/BUSD", nargs='+')
    parser.add_argument("--key", type=str, help="is set to 'date by default", default=["date"], nargs='+')
    parser.add_argument("-t", "--timeframe", type=str, help="exemple : 1m ")
    parser.add_argument("--days", type=int, default="30")
    parser.add_argument("--path", type=str, help="chemin vers les données", 
                        default="./user_data/data/binance")
    parser.add_argument("--col", type=str, default=["all"],
                        nargs='+', help="colonnes qui seront prise en compte ")
    parser.add_argument("--notcol", type=str, default=[],
                        nargs='+', help="colonnes qui ne seront pas prise en compte (if col == 'all') ")
    args = parser.parse_args()
    
    print(os.path.abspath(os.path.curdir))
    path = Path("./user_data/data/binance")
    pair = "BTC/BUSD"
    timeframe = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"] 
    col = ["all"]
    t = timeframe[4]
    
    pair_history = load_pair_history(pair, t, path)
    pair_history.set_index(args.key, inplace=True)
    pair_history = pair_history.loc['2020-12-22':]
    data = populate_indicators(pair_history, t=t)
    plot_acf(data.o)
    plt.show()
    if col == ["all"]:
        columns = sorted(list(set(data.columns) ^ set(args.notcol)))
    else:
        columns = col
    data[columns] = StandardScaler(with_mean=True).fit_transform(data[columns])
    print(data)
    
    ### CALCUL ###

    correlation_do_no = corr_do_no(data.r)
    o_T = o_tau(data.r, 200)
    
    corr_logp_oT = pd.DataFrame()
    corr = pd.Series()
    for i in range(-400, 400):
        corr.loc[i] = data.p.corr(data[f"o[{W_SIZE_VOL}]"].shift(-i))
    corr_logp_oT["corr"] = corr
    
    stats = pd.DataFrame()
    cumul_value = 10
    count, values = np.histogram(data.r, bins = max(len(data.r)//cumul_value, 1)) 
    stats["frequency"] = count
    stats["var_abs"] = values[:-1]
    stats["pdf"] = stats["frequency"] / stats["frequency"].sum()
    stats['cdf'] = 1 - stats["pdf"].cumsum() # prob(x>=)
    
    ###
    
    ### DESSINS COURBE
    figures = []
    trace_data = go.Scatter(x=data.index, y=data.p, name=f"p(t)")
    figures.append((trace_data, 
                    f"p(t)", 
                   "Temps - t", 
                   "p(t)"))
    
    trace_data = go.Scatter(x=data.index, y=data.r, name=f"r(t)")
    figures.append((trace_data, 
                    f"r(t)", 
                   "Temps - t", 
                   "r(t)"))
    
    trace_data = go.Scatter(x=data.index, y=data.cov_r, name=f"Cov(r(t),r(t-1))")
    figures.append((trace_data, 
                    f"Cov(r(t),r(t-1))", 
                   "Temps - t", 
                   "Cov(r(t),r(t-1))"))
    
    trace_data = go.Scatter(x=data.index, y=data.var_r, name=f"Var(r(t))")
    figures.append((trace_data, 
                    f"Var(r(t))", 
                   "Temps - t", 
                   "Var(r(t))"))
    
    trace_data = go.Scatter(x=stats.index, y=stats.pdf, name=f"P(r)")
    figures.append((trace_data, 
                    f"r(t)", 
                   "Variation - r", 
                   "P(r)"))
    
    trace_data = go.Scatter(x=stats.index, y=stats.cdf, name=f"P(r>=a)")
    figures.append((trace_data, 
                    f"r(t)", 
                   "Variation - a", 
                   "P(r>=a)"))
    
    trace_data = go.Scatter(x=data.index, y=data.volume, name=f"v(t)")
    figures.append((trace_data, 
                    f"v(t)", 
                   "Temps - t", 
                   "Volume - v(t)"))
    
    trace_data = go.Scatter(x=data.index, y=data.dv, name=f"dv(t)/dt")
    figures.append((trace_data, 
                    f"dv(t)/dt", 
                   "Temps - t", 
                   "dv(t)/dt"))
    
    trace_data = go.Scatter(x=data.index, y=data[f"o[{W_SIZE_VOL}]"], name=f"o[{W_SIZE_VOL}](t)")
    figures.append((trace_data, 
                    f"o[{W_SIZE_VOL}](t)", 
                   "Temps - t", 
                   f"o[{W_SIZE_VOL}](t)"))
    
    trace_data = go.Scatter(x=data.index, y=data[f"do[{W_SIZE_VOL}]"], name=f"do[{W_SIZE_VOL}](t)/dt")
    figures.append((trace_data, 
                    f"do[{W_SIZE_VOL}](t)/dt", 
                   "Temps - t", 
                   f"do[{W_SIZE_VOL}](t)/dt"))
    
    trace_data = go.Scatter(x=data.index, y=data.o, name=f"o(t)")
    figures.append((trace_data, 
                    f"o(t)", 
                   "Temps - t", 
                   "o(t)"))
    
    trace_data = go.Scatter(x=data.index, y=data.do, name=f"do(t)/dt")
    figures.append((trace_data, 
                    f"do(t)/dt", 
                   "Temps - t", 
                   "do(t)/dt"))
    
    # trace_data = go.Scatter(x=data.index, y=data[f"no[{W_SIZE_VOL}]"], name=f"o[{W_SIZE_VOL}](t+30)")
    # figures.append((trace_data, 
    #                 f"o[{W_SIZE_VOL}](t+30)", 
    #                "Temps - t", 
    #                "o[{W_SIZE_VOL}](t+30)"))
    
    trace_data = go.Scatter(x=data.index, y=data.no, name=f"o(t+30)")
    figures.append((trace_data, 
                    f"o(t+30)", 
                   "Temps - t", 
                   "o(t+30)"))
    
    trace_data = go.Heatmap(z=correlation_do_no, 
                             y=correlation_do_no.index, 
                             x=correlation_do_no.columns,
                             colorscale="spectral", 
                             colorbar=dict(len=.2))
    figures.append((trace_data, 
                    f"Corr(do[doT]/dt, no[noT])", 
                   "doT", 
                   "noT"))

    trace_data = go.Scatter(x=o_T.index, y=o_T["o[T](t)"], name=f"o[T](t)")
    figures.append((trace_data, 
                    f"o[T](t)", 
                   "Fenetre de temps - T", 
                   "o[T](t)"))
    
    
    trace_data = go.Scatter(x=corr_logp_oT.index, y=corr_logp_oT["corr"], name=f"corr[log(p(t)), o(t+T)]")
    figures.append((trace_data, 
                    f"corr[log(p(t)), o(t+T)]", 
                   "Décalage de temps - T", 
                   "corr[log(p(t)), o(t+T)]"))
    ### 
    
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