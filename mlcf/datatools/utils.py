from typing import Callable, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# MLCF modules
from mlcf.datatools.wtseries import WTSeries
from mlcf.datatools.preprocessing import Identity
from mlcf.datatools.indice import add_return
import random

RETURN_COLNAME = "return_close"


def split_pandas(dataframe: pd.DataFrame,
                 prop_snd_elem: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def to_train_val_test(dataframe: pd.DataFrame,
                      prop_tv: float = 0.2,
                      prop_v: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide a dataframe into 3 parts: train part, validation part and test part.

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
    train_data, test_val_data = split_pandas(data, prop_snd_elem=prop_tv)
    val_data, test_data = split_pandas(test_val_data, prop_snd_elem=1-prop_v)
    return train_data, val_data, test_data


def split_in_interval(dataframe: pd.DataFrame,
                      n_interval: int = 1) -> List[pd.DataFrame]:
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
    list_interval: List[pd.DataFrame] = [data.iloc[i*k+min(i, m):(i+1)*k+min(i+1, m)]
                                         for i in range(n_interval)]
    return list_interval


def input_target_data_windows(data_windows: WTSeries,
                              input_width: int,
                              target_width: int) -> Tuple[WTSeries, WTSeries]:
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
    list_in: WTSeries = WTSeries(input_width)
    list_tar: WTSeries = WTSeries(target_width)
    for window in data_windows():
        inp, tar = input_target_data(window, input_width, target_width)
        list_in.add_one_window(inp, ignore_data_empty=True)
        list_tar.add_one_window(tar, ignore_data_empty=True)

    return list_in, list_tar


def input_target_data(dataframe: pd.DataFrame,
                      input_width: int,
                      target_width: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def countinous_value_to_discrete_category(
    dataframe: pd.DataFrame,
    column: str,
    n_bins: int,
    bounds: Tuple[float, float] = None,
    new_columns_name: str = "category"
) -> pd.DataFrame:
    """
    A category refere to a integer label that designates an interval. There are {n_bins} interval,
    divided between the given {bounds}.
    So this function will add a column to the {dataframe} which assign to each row a category
    amoung {n_bins} categories and determined by the values of the given {column}.
    The bounds are by default (-std, std) of the column values. (std = standard deviation)

    Args:
        dataframe (pd.DataFrame): The dataframe on which we will add the column.
        column (str): the column name of the values that are involved
        n_bins (int): the number of category
        bounds (Tuple[float], optional): the bounds gives the lower bound of the lowest interval,
        and the higher bound of the higher interval. Defaults to (-std, std).
        new_columns_name (str, optional): The name of the new category column.
        Defaults to "category".

    Returns:
        pd.DataFrame: The modified dataframe
    """
    data = dataframe.copy()
    if not bounds:
        mean = data[column].mean()
        bounds = (mean - data[column].std(), mean + data[column].std())
    bins = np.linspace(bounds[0], bounds[1], n_bins)
    data[new_columns_name] = pd.cut(data[column], bins).cat.codes
    data[new_columns_name] += 1
    data.loc[
        (data[new_columns_name] == 0) &
        (data[column] > bounds[1]),
        new_columns_name
    ] = n_bins
    return data


def balanced_category_tag(
    dataframe: pd.DataFrame,
    category_column: str,
    tag_col_name: str = "tag",
    max_count: int = None,
    sample_function: Callable = random.sample
) -> pd.DataFrame:
    """
    Will tag True or False a row in order to have approximately the same amount of row tagged True
    for each category. Then if you select all the row tagged True and you calculate an histogram on
    it, the histogram should be flattens (more uniform).

    Args:
        dataframe (pd.DataFrame): the dataframe, should have a category column
        category_column (str): the category column give from what category a row belong
        tag_col_name (str, optional): The column name of the tag column. Defaults to "tag".
        max_count (int, optional): You can fix the max amount of row to tag to True
        for each category. Defaults to None.
        sample_function (Callable, optional): The function to sample the rows to tag True.
        By default it is random.
        Sample function spec:(
        In: a list of index, and the size of the sample to return,
        Out: return a list of index.)
        Defaults to random.sample.

    Returns:
        pd.DataFrame: The tagged dataframe
    """
    data = dataframe.copy()
    idx_cat = data[category_column].value_counts().index
    value_count = data[category_column].value_counts()
    if not max_count:
        max_count = np.mean([value_count[min(idx_cat)], value_count[max(idx_cat)]]).astype(int)
    data[tag_col_name] = True
    for idx in sorted(value_count.index):
        if value_count[idx] > max_count:
            data.loc[
                sorted(sample_function(
                    list(data[data[category_column] == idx].index),
                    k=value_count[idx]-max_count)
                ), tag_col_name
            ] = False
    return data


def balance_category_by_tagging(
        intervals_data: List[pd.DataFrame],
        n_category: int,
        target_width: int,
        offset: int,
        bounds: Tuple[float, float] = None,
        max_count: int = None,
        cat_col_name: str = "category",
        tag_col_name: str = "tag",
        sample_function: Callable = random.sample) -> Tuple[List[pd.DataFrame], Union[str, None]]:
    """From a list of OHLCV data, this function will assign a category of the returns of the
    close value. The return is compute at a time 't' as follow :
        dclose(t) = ln{close(t)} - ln{close(t-target_width-offset)}
    A number of return interval is compute and for each interval a label category is given.
    Then the function will tag each row True or False in order to balance the number of row tagged
    True in each category.
    The effect of this process allows if you select each row tagged True, to flattens the histogram
    of the categories.

    Args:
        intervals_data (List[pd.DataFrame]): list of OHLCV dataframe
        n_category (int): the number of categories(intervals) to compute
        target_width (int): the width of the target
        offset (int): the width of the offset
        bounds (Tuple[float], optional): the bounds gives the lower bound of the lowest interval
        max_count (int, optional) = You can fix the max amount of row to tag to True
        for each category. Defaults to None. (Optional)
        cat_col_name (str, optional): the column name of the category. Defaults to "category".
        tag_col_name (str, optional): the column name of the tag. Defaults to "tag".
        sample_function (Callable, optional): The function to sample the rows to tag True.
            By default it is random.
            Sample function spec:(
            In: a list of index, and the size of the sample to return,
            Out: return a list of index.)
            Defaults to random.sample.
    Returns:
        List[pd.DataFrame]: The list of categorized and tagged OHLCV data
    """
    if n_category > 1:
        for idx, _ in enumerate(intervals_data):
            intervals_data[idx] = add_return(
                intervals_data[idx],
                offset=target_width+offset,
                colname=RETURN_COLNAME,
                dropna=True)
            intervals_data[idx] = countinous_value_to_discrete_category(
                intervals_data[idx],
                column=RETURN_COLNAME,
                n_bins=n_category,
                bounds=bounds,
                new_columns_name=cat_col_name
            )
            intervals_data[idx] = balanced_category_tag(
                intervals_data[idx],
                cat_col_name,
                max_count=max_count,
                tag_col_name=tag_col_name,
                sample_function=sample_function)
        return intervals_data, tag_col_name
    return intervals_data, None


def train_val_test_list_data(
    list_interval_data_df: List[pd.DataFrame],
    prop_tv: float,
    prop_v: float
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:

    splited_interval_data: Tuple[List[pd.DataFrame],
                                 List[pd.DataFrame],
                                 List[pd.DataFrame]] = ([], [], [])
    for interval_data_df in list_interval_data_df:
        train, val, test = to_train_val_test(
            interval_data_df,
            prop_tv=prop_tv,
            prop_v=prop_v)
        splited_interval_data[0].append(train)
        splited_interval_data[1].append(val)
        splited_interval_data[2].append(test)

    return splited_interval_data


def generate_windows_from_splited_interval_data(
    splited_interval_data: Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]],
    input_width: int,
    offset: int,
    target_width: int,
    window_step: int,
    preprocess=Identity,
    features: List[str] = None,
    tag_name: str = None,
) -> Tuple[WTSeries, WTSeries, WTSeries, WTSeries, WTSeries, WTSeries]:

    window_width: int = input_width + offset + target_width

    train_input: WTSeries = WTSeries(window_width=input_width, window_step=window_step)
    train_target: WTSeries = WTSeries(window_width=target_width, window_step=window_step)
    val_input: WTSeries = WTSeries(window_width=input_width, window_step=window_step)
    val_target: WTSeries = WTSeries(window_width=target_width, window_step=window_step)
    test_input: WTSeries = WTSeries(window_width=input_width, window_step=window_step)
    test_target: WTSeries = WTSeries(window_width=target_width, window_step=window_step)

    for train, val, test in zip(*splited_interval_data):  # for each interval

        train_prep = preprocess(
            WTSeries(
                raw_data=train,
                window_width=window_width,
                window_step=window_step,
                tag_name=tag_name,
                features=features
            )
        )

        val_prep = preprocess(
            WTSeries(
                raw_data=val,
                window_width=window_width,
                window_step=window_step,
                features=features
            )
        )

        test_prep = preprocess(
            WTSeries(
                raw_data=test,
                window_width=window_width,
                window_step=window_step,
                features=features
            )
        )

        train_data: WTSeries = train_prep()
        val_data: WTSeries = val_prep()
        test_data: WTSeries = test_prep()

        train_input_tmp, train_target_tmp = input_target_data_windows(
            train_data,
            input_width,
            target_width
        )
        val_input_tmp, val_target_tmp = input_target_data_windows(
            val_data,
            input_width,
            target_width
            )
        test_input_tmp, test_target_tmp = input_target_data_windows(
            test_data,
            input_width,
            target_width
        )

        train_input.add_window_data(train_input_tmp, ignore_data_empty=True)
        train_target.add_window_data(train_target_tmp, ignore_data_empty=True)
        val_input.add_window_data(val_input_tmp, ignore_data_empty=True)
        val_target.add_window_data(val_target_tmp, ignore_data_empty=True)
        test_input.add_window_data(test_input_tmp, ignore_data_empty=True)
        test_target.add_window_data(test_target_tmp, ignore_data_empty=True)

    return (train_input, train_target, val_input, val_target, test_input, test_target)


def standardize(
    splited_interval_data:
        Tuple[
            List[pd.DataFrame],
            List[pd.DataFrame],
            List[pd.DataFrame]
        ],
    list_to_std: List[str]
):
    if list_to_std:
        sc = StandardScaler()
        for train, val, test in zip(*splited_interval_data):
            if not train.empty:
                sc.partial_fit(train[list_to_std])

        for train, val, test in zip(*splited_interval_data):
            if not train.empty:
                train[list_to_std] = sc.transform(train[list_to_std])

            if not val.empty:
                val[list_to_std] = sc.transform(val[list_to_std])

            if not test.empty:
                test[list_to_std] = sc.transform(test[list_to_std])

    return splited_interval_data


def build_forecast_ts_training_dataset(
    dataframe: pd.DataFrame,
    input_width: int,
    target_width: int = 1,
    offset: int = 0,
    window_step: int = 1,
    n_interval: int = 1,
    prop_tv: float = 0.2,
    prop_v: float = 0.3,
    preprocess=Identity,
    n_category: int = 0,
    bounds: Tuple[float, float] = None,
    max_count: int = None,
    list_to_std: List[str] = []
) -> Tuple[WTSeries, WTSeries, WTSeries, WTSeries, WTSeries, WTSeries]:
    """ From a time serie dataframe, build a forecast training dataset:
    -> ({n_interval} > 1): divide the dataframe in {n_interval} intervals
    -> split the dataframe in train, validation and test part
    -> balance the train_data by categorizing the return in n_category and tagging category True
    or False in order to have approximately the same amount in each return category. (flatten the
    return histogram if we select rows tagged True)
    -> windows the data with a window size of ({input_width} + {target_width} + {offset}) and
    a window_step of {window_step} and select windows if the last row of the window is tagged True
    -> make the X and y (input and target) parts given the {input_width} and {target_width}
    -> make the lists of train part inputs, train part targets, validation part inputs, validation
    part targets, test part inputs and test part targets

    Args:
        dataframe (pd.DataFrame): The dataframe (time serie)
        input_width (int): The input size in a window of the data
        target_width (int, optional): The tarsplited_interval_dataion of the union of
        [test and validation] part.
        Defaults to 0.2.
        prop_v (float, optional): the proportion of validation in the union of
        [test and validation] part. Defaults to 0.4.
        preprocess (PreProcess, optional): is a preprocessing function taking a WTSeries in input.
        default to Identity
        n_category (int): The number of return categories. (Default to 0)
        bounds (Tuple[float]): The bounds of the returns to categories. (Optional)
        max_count (int): the max value tagged true in each category. (Optional)

    Returns:
        Tuple[WTSeries,
              WTSeries,
              WTSeries,
              WTSeries,
              WTSeries,
              WTSeries],
            the lists of train part inputs, train part targets, validation part inputs,
            validation part targets,  test part inputs and test part targets
    """
    data = dataframe.copy()
    features = list(data.columns)
    # Divide data in N interval
    list_interval_data_df: List[pd.DataFrame] = split_in_interval(data, n_interval=n_interval)

    # Split each interval data in train val test
    splited_interval_data: Tuple[
        List[pd.DataFrame],
        List[pd.DataFrame],
        List[pd.DataFrame]
    ] = train_val_test_list_data(list_interval_data_df, prop_tv=prop_tv, prop_v=prop_v)

    splited_interval_data = standardize(splited_interval_data, list_to_std)

    train_intervals_tagged, tag_name = balance_category_by_tagging(
        splited_interval_data[0],
        n_category,
        target_width=target_width,
        offset=offset,
        bounds=bounds,
        max_count=max_count)
    splited_interval_data = (
        train_intervals_tagged,
        splited_interval_data[1],
        splited_interval_data[2])

    # Generate inputs and targets windows
    windows = generate_windows_from_splited_interval_data(
        splited_interval_data,
        input_width=input_width,
        offset=offset,
        target_width=target_width,
        window_step=window_step,
        preprocess=preprocess,
        tag_name=tag_name,
        features=features
    )
    (train_input, train_target, val_input, val_target, test_input, test_target) = windows

    return (train_input, train_target, val_input, val_target, test_input, test_target)
