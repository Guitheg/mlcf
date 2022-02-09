from typing import List, Tuple
import pandas as pd
import random

### CG-RBI modules ###
from CGrbi.datatools.wtseries import WTSeries
from CGrbi.datatools.preprocessing import Identity, WTSeriesPreProcess

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

    k, m = divmod(len(data), n_interval)
    list_interval : List[pd.DataFrame] = [data.iloc[i*k+min(i, m):(i+1)*k+min(i+1, m)] 
                                          for i in range(n_interval)]
    return list_interval

def input_target_data_windows(data_windows : WTSeries,
                            input_width : int,
                            target_width : int) -> Tuple[WTSeries, WTSeries]:
    """Given a list of windows (of dataframe), return list of the input and the target parts 
    of these windows.

    Args:
        data_windows (List[pd.DataFrame]): The windowed data (dataframe)
        input_width (int): input width in the window
        target_width (int): target width in the window

    Returns:
        Tuple[List[pd.DataFrame], List[pd.DataFrame]]: The list of inputpars, 
        and the list of target parts
    """
    list_in : WTSeries = WTSeries(input_width)
    list_tar : WTSeries = WTSeries(target_width)
    for window in data_windows:
        inp, tar = input_target_data(window, input_width, target_width)
        list_in.add_one_window(inp)
        list_tar.add_one_window(tar)
    
    return list_in, list_tar

def input_target_data(dataframe : pd.DataFrame,
               input_width : int,
               target_width : int) -> Tuple[pd.DataFrame, pd.DataFrame] :
    """From a window (a dataframe), return the list of the input and the target part given the
    {input_width} and the {target_width}.

    Args:
        dataframe (pd.DataFrame): The dataframe
        input_width (int): the size of the input
        target_width (int): the size of the target

    Raises:
        Exception: if the input and target sizes are greater than the window size

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: the input part and the target part
    """
    data = dataframe.copy()
    if len(data) == 0:
        return pd.DataFrame(columns=data.columns), pd.DataFrame(columns=data.columns)
    if len(data) < input_width + target_width:
        raise Exception("Input width and Label width must not be greater than window size")
    input_data = data.iloc[:input_width]
    target_data = data.iloc[-target_width:]
    return input_data, target_data

def build_forecast_ts_training_dataset(dataframe : pd.DataFrame,
                                       input_width : int,
                                       target_width : int = 1,
                                       offset : int = 0,
                                       window_step : int = 1,
                                       n_interval : int = 1,
                                       test_val_prop : float = 0.2,
                                       val_prop : float = 0.4,
                                       do_shuffle : bool = False,
                                       preprocess : WTSeriesPreProcess = Identity,
                                       ) -> Tuple[List[pd.DataFrame],
                                                  List[pd.DataFrame],
                                                  List[pd.DataFrame],
                                                  List[pd.DataFrame],
                                                  List[pd.DataFrame],
                                                  List[pd.DataFrame]]:
    """ From a time serie dataframe, build a forecast training dataset:
    -> ({n_interval} > 1) : divide the dataframe in {n_interval} intervals
    -> split the dataframe in train, validation and test part
    -> windows the data with a window size of ({input_width} + {target_width} + {offset}) and 
    a window_step of {window_step}
    -> make the X and y (input and target) parts given the {input_width} and {target_width}
    -> make the lists of train part inputs, train part targets, validation part inputs, validation
    part targets, test part inputs and test part targets
    -> ({do_shuffle} is True) : shuffle all the lists

    Args:
        dataframe (pd.DataFrame): The dataframe (time serie)
        input_width (int): The input size in a window of the data
        target_width (int, optional): The target size in a window of the data. Defaults to 1.
        offset (int, optional): the offset size between the input width and the target width. 
        Defaults to 0.
        window_step (int, optional): to select a window every window_step. Defaults to 1.
        n_interval (int, optional): the number of splited intervals. Defaults to 1.
        test_val_prop (float, optional): the proportion of the union of [test and validation] part.
        Defaults to 0.2.
        val_prop (float, optional): the proportion of validation in the union of 
        [test and validation] part. Defaults to 0.4.
        do_shuffle (bool, optional) : if True, do a shuffle on the data. Default to False.
        preprocess (PreProcess, optional) : is a preprocessing function taking a WTSeries in input.
        default to Identity

    Returns:
        Tuple[
            List[pd.DataFrame], 
            List[pd.DataFrame], 
            List[pd.DataFrame],
            List[pd.DataFrame],
            List[pd.DataFrame], 
            List[pd.DataFrame]]: 
            the lists of train part inputs, train part targets, validation part inputs, 
            validation part targets,  test part inputs and test part targets
    """
    data = dataframe.copy()
    # Divide data in N interval
    list_interval_data_df : List[pd.DataFrame] = split_in_interval(data, n_interval=n_interval)
    
    # Split each interval data in train val test
    splited_interval_data : Tuple[List[pd.DataFrame], 
                                     List[pd.DataFrame], 
                                     List[pd.DataFrame]] = ([],[],[])
    for interval_data_df in list_interval_data_df:
        train, val, test  = to_train_val_test(interval_data_df, 
                                              test_val_prop=test_val_prop, 
                                              val_prop=val_prop)
        splited_interval_data[0].append(train)
        splited_interval_data[1].append(val)
        splited_interval_data[2].append(test)

    # Generate windowed data and targets
    window_width : int = input_width + offset + target_width
    train_input : WTSeries = WTSeries(window_width=input_width, window_step=window_step)
    train_target : WTSeries = WTSeries(window_width=target_width, window_step=window_step)
    val_input : WTSeries = WTSeries(window_width=input_width, window_step=window_step)
    val_target : WTSeries = WTSeries(window_width=target_width, window_step=window_step)
    test_input : WTSeries = WTSeries(window_width=input_width, window_step=window_step)
    test_target : WTSeries = WTSeries(window_width=target_width, window_step=window_step)
    
    for train, val, test in zip(*splited_interval_data): # for each interval
        train_prep = preprocess(WTSeries(data=train, 
                                         window_width=window_width, 
                                         window_step=window_step))
        val_prep = preprocess(WTSeries(data=val, 
                                       window_width=window_width, 
                                       window_step=window_step))
        test_prep = preprocess(WTSeries(data=test, 
                                        window_width=window_width, 
                                        window_step=window_step))
        train_data = train_prep()
        val_data = val_prep()
        test_data = test_prep()
        
        train_input_tmp, train_target_tmp = input_target_data_windows(train_data, 
                                                                      input_width, 
                                                                      target_width)
        val_input_tmp, val_target_tmp = input_target_data_windows(val_data, 
                                                                      input_width, 
                                                                      target_width)
        test_input_tmp, test_target_tmp = input_target_data_windows(test_data, 
                                                                      input_width, 
                                                                      target_width)
        
        train_input.merge_window_data(train_input_tmp)
        train_target.merge_window_data(train_target_tmp)
        val_input.merge_window_data(val_input_tmp)
        val_target.merge_window_data(val_target_tmp)
        test_input.merge_window_data(test_input_tmp)
        test_target.merge_window_data(test_target_tmp)

    if do_shuffle:
        train_input.make_common_shuffle(train_target)
        val_input.make_common_shuffle(val_target)
        test_input.make_common_shuffle(test_target)
        
    return (train_input, train_target, val_input, 
            val_target, test_input, test_target)


