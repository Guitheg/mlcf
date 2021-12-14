import os
from re import T
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


def process_pca(data_df: pd.DataFrame, n_components: int, plot: bool = False):
    data = np.array(data_df)[:,1:]  # to erase timestamp
    data = StandardScaler().fit_transform(data)

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

def main():
    print(os.path.abspath(os.path.curdir))
    BTC_BUSD_5m = load_pair_history("BTC/BUSD", "1m", Path("./user_data/data/binance"))
    pca_result = process_pca(BTC_BUSD_5m, n_components=2, plot=True)
    
if __name__ == "__main__":
    main()