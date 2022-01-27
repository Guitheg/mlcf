from pathlib import Path
from typing import List, Tuple
from freqtrade.data.history.history_utils import load_pair_history

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sqlalchemy import column

def split_pandas(dataframe : pd.DataFrame, 
                 prop_snd_elem : float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame] :
    data = dataframe.copy()
    times = sorted(data.index.values)
    second_part = sorted(data.index.values)[-int(prop_snd_elem*len(times))]
    second_data = data[(data.index >= second_part)]
    first_data = data[(data.index < second_part)]
    return first_data, second_data

def to_train_val_test(dataframe : pd.DataFrame, 
                      test_val : float = 0.2,
                      val : float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    data = dataframe.copy()
    train_data, test_val_data = split_pandas(data, prop_snd_elem = test_val)
    test_data, val_data = split_pandas(test_val_data, prop_snd_elem = val)
    return train_data, val_data, test_data

def divide_in_period(dataframe : pd.DataFrame, 
                     n_period : int = 1) -> List[pd.DataFrame]:
    data = dataframe.copy()
    window_width : int = len(data.index)//n_period
   
    return window_data(data, window_width, step=window_width)

def window_data(dataframe : pd.DataFrame, 
                window_width : int,
                step : int = 1) -> List[pd.DataFrame]:
    data = dataframe.copy()
    n_windows = ((len(data.index)-window_width) // step) + 1
    n_columns = len(data.columns)
    
    # Slid window on all data
    windowed_data : np.ndarray = sliding_window_view(data, 
                                        window_shape=(window_width, len(data.columns)))

    # Take data every step
    windowed_data = windowed_data[::step]
    
    # Reshape windowed data
    windowed_data_shape : Tuple[int, int, int]= (n_windows, window_width, n_columns)
    windowed_data = np.reshape(windowed_data, newshape = windowed_data_shape)
    
    # Make list of dataframe
    list_data : List[pd.DataFrame] = []
    for idx in range(n_windows):
        list_data.append(
            pd.DataFrame(windowed_data[idx], 
                        index = data.index[idx*step : (idx*step)+window_width],
                        columns = data.columns)
            )
    return list_data

def input_label_data_windows(data_windows : List[pd.DataFrame],
                            input_width : int,
                            label_width : int) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    list_in : List[pd.DataFrame] = []
    list_lab : List[pd.DataFrame] = []
    for window in data_windows:
        inp, lab = input_label_data(window, input_width, label_width)
        list_in.append(inp)
        list_lab.append(lab)
    
    return list_in, list_lab

def input_label_data(dataframe : pd.DataFrame,
               input_width : int,
               label_width : int) -> Tuple[pd.DataFrame, pd.DataFrame] :
    data = dataframe.copy()
    if len(data) < input_width + label_width:
        raise Exception("Input width and Label width must not be greater than window size")
    input_data = data.iloc[:input_width]
    label_data = data.iloc[-label_width:]
    return input_data, label_data

def build_forecast_ts_training_dataset(dataframe : pd.DataFrame,
                                       input_width : int = 3,
                                       offset : int = 0,
                                       label_width : int = 1,
                                       step : int = 1,
                                       n_period : int = 1,
                                       test_val_prop : float = 0.2,
                                       val_prop : float = 0.4
                                       ) -> Tuple[List[pd.DataFrame],
                                                  List[pd.DataFrame],
                                                  List[pd.DataFrame],
                                                  List[pd.DataFrame],
                                                  List[pd.DataFrame],
                                                  List[pd.DataFrame]]:
    """Build a forecast time series training dataset

    Args:
        dataframe (pd.DataFrame): [1-D Time series]
        past_window (int, optional): Defaults to 20.
        fututre_window (int, optional): Defaults to 20.

    Returns:
        pd.DataFrame: [description]
    """
    data = dataframe.copy()
    # Divide data in N period
    list_period_data_df : List[pd.DataFrame] = divide_in_period(data, n_period=n_period)
    
    # Split each period data in train val test
    list_splited_period_data_df : List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for period_data_df in list_period_data_df:
        list_splited_period_data_df.append(
            to_train_val_test(period_data_df, test_val=test_val_prop, val=val_prop) 
        )
    
    # Generate windowed data and labels
    window_size : int = input_width + offset + label_width
    list_train_input : List[pd.DataFrame] = []
    list_train_label : List[pd.DataFrame] = []
    list_val_input : List[pd.DataFrame] = []
    list_val_label : List[pd.DataFrame] = []
    list_test_input : List[pd.DataFrame] = []
    list_test_label : List[pd.DataFrame] = []
    for train, val, test in list_splited_period_data_df:
        train_input, train_label = input_label_data_windows(
            window_data(train, window_size, step=step), 
            input_width, 
            label_width)
        val_input, val_label = input_label_data_windows(
            window_data(val, window_size, step=step),
            input_width, 
            label_width)
        test_input, test_label = input_label_data_windows(
            window_data(test, window_size, step=step),
            input_width, 
            label_width)
        list_train_input.extend(train_input)
        list_train_label.extend(train_label)
        list_val_input.extend(val_input)
        list_val_label.extend(val_label)
        list_test_input.extend(test_input)
        list_test_label.extend(test_label)

    return (list_train_input, 
            list_train_label, 
            list_val_input, 
            list_val_label, 
            list_test_input, 
            list_test_label)


def main():
    path = Path("./user_data/data/binance")
    pair = "BTC/BUSD"
    timeframe = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"] 
    col = ["all"]
    t = timeframe[4]
    
    pair_history = load_pair_history(pair, t, path)
    # pair_history.set_index("date", inplace=True)
    dataframe = pair_history[["close", "open", "volume"]]
    x,y,_,_,_,_ =  build_forecast_ts_training_dataset(dataframe)
    print(x[0])
    print(y[0])
    
    print(dataframe)
    # data_ts.plot()


if __name__ == "__main__":
    main()