from pathlib import Path
from freqtrade.data.history.history_utils import load_pair_history

import pandas as pd
import numpy as np



def build_forecast_ts_training_dataset(dataframe : pd.DataFrame, 
                                       past_window : int = 20,
                                       future_window : int = 20) -> pd.DataFrame:
    """Build a forecast time series training dataset

    Args:
        dataframe (pd.DataFrame): [1-D Time series]
        past_window (int, optional): Defaults to 20.
        fututre_window (int, optional): Defaults to 20.

    Returns:
        pd.DataFrame: [description]
    """
    data = dataframe.copy()
    training_dataset = pd.DataFrame()
    training_dataset = data.rolling(past_window+future_window)
    print(training_dataset)

def main():
    path = Path("./user_data/data/binance")
    pair = "BTC/BUSD"
    timeframe = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"] 
    col = ["all"]
    t = timeframe[4]
    
    pair_history = load_pair_history(pair, t, path)
    
    pair_history.set_index("date", inplace=True)
    print(pair_history)

    # data_ts.plot()


if __name__ == "__main__":
    main()