import os
import argparse
from pathlib import Path
from functools import reduce

from freqtrade.data.history.history_utils import load_pair_history


from pandas_ta.overlap.ichimoku import ichimoku
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import pandas as pd

import plotly
import plotly.express as px
import plotly.graph_objects as go


def populate_indicators(data : pd.DataFrame) -> pd.DataFrame:
    dataframe = data.copy()
    
    dataframe.dropna(inplace=True)
    return dataframe

def process(data):
    
    return data
    

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
    parser.add_argument("--notcol", type=str, default=["date"],
                        nargs='+', help="colonnes qui ne seront pas prise en compte (if col == 'all') ")
    args = parser.parse_args()
    
    print(os.path.abspath(os.path.curdir))
    path = Path(args.path)
    pairs = args.pair_names
    pair_histories = []
    data = pd.DataFrame()
    if os.path.isdir(path):
        for pair in pairs:
            try:
                command = "freqtrade download-data --exchange binance"+ \
                    f" --pairs '{pair}' --days '{args.days}' --timeframe '{args.timeframe}'"
                print(command)
                os.system(command)
            except:
                raise Exception("Le téléchargement des données a échoué")
            
            pair_history = load_pair_history(pair, args.timeframe, path)
            pair_history = populate_indicators(pair_history)
            prefix = f"{pair}_"
            pair_history = pair_history.rename(columns = lambda col: f"{prefix}{col}" 
                                if col not in tuple(args.key)
                                else col)
            pair_histories.append(pair_history)
            
        data = reduce(lambda  left,right: pd.merge(left,right,on=args.key,
                                            how='outer'), pair_histories)
                
        if args.col == ["all"]:
            columns = sorted(list(set(data.columns) ^ set(args.notcol)))
        else:
            columns = args.col
    else:
        raise Exception(f"Le chemin est inconnu")
    
    print(data)
    
    figures = []
    
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
    
    if len(figures) > 1:    
        window.layout.xaxis["title"] = figures[0][2]
        window.layout.yaxis["title"] = figures[0][3]
        for trace in range(1, len(figures)):
            window.layout[f"xaxis{trace+1}"]["title"] = figures[trace][2]
            window.layout[f"yaxis{trace+1}"]["title"] = figures[trace][3]
    if len(figures) > 0:
        window.show()
    
if __name__ == "__main__":
    main()