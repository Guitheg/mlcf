import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
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
                                      columns=[f"C{i}" for i in range(n_components)])
    if plot == True:
        if n_components == 2:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(1,1,1)
            ax.set_xlabel(pcd_data_df_result.columns[0])
            ax.set_ylabel(pcd_data_df_result.columns[1])
            ax.set_title('2 component PCA', fontsize=20)
            ax.scatter(pcd_data_df_result[pcd_data_df_result.columns[0]], 
                       pcd_data_df_result[pcd_data_df_result.columns[1]])
            ax.grid()
            plt.show()
        else:
            raise NotImplementedError("n_component =/= 2 not implemented")
    
    return pcd_data_df_result

def main():
    print(os.path.abspath(os.path.curdir))
    BTC_BUSD_5m = np.array(load_pair_history("BTC/BUSD", "5m", Path("./user_data/data/binance")))[:,1:]
    pca_result = process_pca(BTC_BUSD_5m, n_components=2, plot=True)
    
if __name__ == "__main__":
    main()