
from typing import List, Tuple

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import random

def split_pandas(dataframe : pd.DataFrame, 
                 prop_snd_elem : float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame] :
    """Split (from indexes) a dataframe in two dataframes which keep the same columns.
    The {prop_snd_elem} is the proportion of row for the second element.

    Args:
        dataframe (pd.DataFrame): The dataframe we want to split in two
        prop_snd_elem (float, optional): The proportion of row of the second elements in percentage. 
        Defaults to 0.5.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The first and second part of the split
    """
    data = dataframe.copy()
    if len(data) == 0:
        return pd.DataFrame(columns=data.columns), pd.DataFrame(columns=data.columns)
    if prop_snd_elem == 0.0:
        return data, pd.DataFrame(columns=data.columns)
    elif prop_snd_elem == 1.0:
        return pd.DataFrame(columns=data.columns), data
    elif prop_snd_elem < 0.0 or prop_snd_elem > 1.0:
        raise Exception("prop_sn_elem should be between 0 and 1")
    else:
        times = sorted(data.index)
        second_part = times[-int(prop_snd_elem*len(times))]
        second_data = data[(data.index >= second_part)]
        first_data = data[(data.index < second_part)]
        return first_data, second_data

def to_train_val_test(dataframe : pd.DataFrame, 
                      test_val_prop : float = 0.2,
                      val_prop : float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    """Divide a dataframe into 3 parts : train part, validation part and test part.

    Args:
        dataframe (pd.DataFrame): dataframe we want to split in 3
        test_val (float, optional): the proportion of the union of the [test and validation] part. 
        Defaults to 0.2.
        val (float, optional): the proportion of the validation part in the union of [test and 
        validation] part. Defaults to 0.5.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Respectively the train, val and test part
    """
    data = dataframe.copy()
    train_data, test_val_data = split_pandas(data, prop_snd_elem = test_val_prop)
    test_data, val_data = split_pandas(test_val_data, prop_snd_elem = val_prop)
    return train_data, val_data, test_data

def split_in_interval(dataframe : pd.DataFrame, 
                     n_interval : int = 1) -> List[pd.DataFrame]:
    """split (in row) the dataframe in {n_interval} intervals 

    Args:
        dataframe (pd.DataFrame): the dataframe we want to split
        n_interval (int, optional): the number of interval. Defaults to 1.

    Returns:
        List[pd.DataFrame]: The list of intervals
    """
    data = dataframe.copy()
    if len(data) == 0:
        return [pd.DataFrame(columns=data.columns)]
    window_width : int = len(data.index)//n_interval
   
    return window_data(data, window_width, step=window_width)

def window_data(dataframe : pd.DataFrame, 
                window_width : int,
                step : int = 1) -> List[pd.DataFrame]:
    """Window the data given a {window_width} and a {step}.

    Args:
        dataframe (pd.DataFrame): The dataframe we want to window
        window_width (int): the windows size
        step (int, optional): the step between each window. Defaults to 1.

    Returns:
        List[pd.DataFrame]: The list of created windows of size {window_width} and 
        selected every step {step}
    """
    data = dataframe.copy()
    if len(data) == 0:
        return [pd.DataFrame(columns=data.columns)]
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
    """Given a list of windows (of dataframe), return list of the input and the label parts 
    of these windows.

    Args:
        data_windows (List[pd.DataFrame]): The windowed data (dataframe)
        input_width (int): input width in the window
        label_width (int): label width in the window

    Returns:
        Tuple[List[pd.DataFrame], List[pd.DataFrame]]: The list of inputpars, 
        and the list of label parts
    """
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
    """From a window (a dataframe), return the list of the input and the label part given the
    {input_width} and the {label_width}.

    Args:
        dataframe (pd.DataFrame): The dataframe
        input_width (int): the size of the input
        label_width (int): the size of the label

    Raises:
        Exception: if the input and label sizes are greater than the window size

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: the input part and the label part
    """
    data = dataframe.copy()
    if len(data) == 0:
        return pd.DataFrame(columns=data.columns), pd.DataFrame(columns=data.columns)
    if len(data) < input_width + label_width:
        raise Exception("Input width and Label width must not be greater than window size")
    input_data = data.iloc[:input_width]
    label_data = data.iloc[-label_width:]
    return input_data, label_data

def build_forecast_ts_training_dataset(dataframe : pd.DataFrame,
                                       input_width : int,
                                       label_width : int = 1,
                                       offset : int = 0,
                                       step : int = 1,
                                       n_interval : int = 1,
                                       test_val_prop : float = 0.2,
                                       val_prop : float = 0.4,
                                       do_shuffle : bool = False,
                                       ) -> Tuple[List[pd.DataFrame],
                                                  List[pd.DataFrame],
                                                  List[pd.DataFrame],
                                                  List[pd.DataFrame],
                                                  List[pd.DataFrame],
                                                  List[pd.DataFrame]]:
    """ From a time serie dataframe, build a forecast training dataset:
    -> ({n_interval} > 1) : divide the dataframe in {n_interval} intervals
    -> split the dataframe in train, validation and test part
    -> windows the data with a window size of ({input_width} + {label_width} + {offset}) and 
    a step of {step}
    -> make the X and y (input and label) parts given the {input_width} and {label_width}
    -> make the lists of train part inputs, train part labels, validation part inputs, validation
    part labels, test part inputs and test part labels
    -> ({do_shuffle} is True) : shuffle all the lists

    Args:
        dataframe (pd.DataFrame): The dataframe (time serie)
        input_width (int): The input size in a window of the data
        label_width (int, optional): The label size in a window of the data. Defaults to 1.
        offset (int, optional): the offset size between the input width and the label width. 
        Defaults to 0.
        step (int, optional): to select a window every step. Defaults to 1.
        n_interval (int, optional): the number of splited intervals. Defaults to 1.
        test_val_prop (float, optional): the proportion of the union of [test and validation] part.
        Defaults to 0.2.
        val_prop (float, optional): the proportion of validation in the union of 
        [test and validation] part. Defaults to 0.4.
        do_shuffle (bool, optional) : if True, do a shuffle on the data

    Returns:
        Tuple[
            List[pd.DataFrame], 
            List[pd.DataFrame], 
            List[pd.DataFrame],
            List[pd.DataFrame],
            List[pd.DataFrame], 
            List[pd.DataFrame]]: 
            the lists of train part inputs, train part labels, validation part inputs, 
            validation part labels,  test part inputs and test part labels
    """
    data = dataframe.copy()
    # Divide data in N period
    list_period_data_df : List[pd.DataFrame] = split_in_interval(data, n_interval=n_interval)
    
    # Split each period data in train val test
    list_splited_period_data_df : List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for period_data_df in list_period_data_df:
        list_splited_period_data_df.append(
            to_train_val_test(period_data_df, test_val_prop=test_val_prop, val_prop=val_prop) 
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

    if do_shuffle:
        list_train = list(zip(list_train_input, list_train_label))
        list_validation = list(zip(list_val_input, list_val_label))
        list_test = list(zip(list_test_input, list_test_label))
        random.shuffle(list_train)
        random.shuffle(list_validation)
        random.shuffle(list_test)
        list_train_input, list_train_label = zip(*list_train)
        list_val_input, list_val_label = zip(*list_validation)
        list_test_input, list_test_label = zip(*list_test)
        
        
    return (list_train_input, list_train_label, list_val_input, 
            list_val_label, list_test_input, list_test_label)
