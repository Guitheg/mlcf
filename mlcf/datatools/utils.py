"""Utilitaries module.

Provide some tools to perform processing on data frame.
"""

from copy import copy
from typing import List, Tuple, Union
import pandas as pd
import numpy as np

# TODO (doc) review

__all__ = [
    "binarize",
    "labelize",
    "split_pandas",
    "split_train_val_test"
]


def binarize(
    data: pd.DataFrame,
    column: str,
    labels: Union[Tuple, List] = None,
    sep_value: Union[float, int] = 0.0,
    include_sep: bool = True,
    label_col_name: str = None
) -> pd.DataFrame:
    """
    Add a {label_col_name} column to a dataframe containing the binary label of a feature.
    We can choose the value of the separator as follows: any value in the column less than the
    {sep_value} belongs to the first class and any value greater than or equal to the {sep_value}
    belongs to the second class. The upper or equal condition can be changed to a simple upper
    condition if we set {include_sep} to False.
    The new column will have the name {label_col_name}, by default it will be: '{column}_label'.

    Args:
        data (pd.DataFrame): The dataframe

        column (str): The column name of the feature.

        labels (Union[Tuple, List], optional): A tuple or list of two element (the labels names).
            If None then the two elements will be (0, 1). Defaults to None.

        sep_value (Union[float, int], optional): The value which determine the condition to belong
            to the first or the second class. If value are less than {sep_value} then it belongs to
            the first class else it belongs to the second class. Defaults to 0.0.

        include_sep (bool, optional): If False, the value should be less or equal to {sep_value}
             belong to the first class. Defaults to True.

        label_col_name (str, optional): The column name of the new label column. By default, the
            name is : '{column}_label'. Defaults to None.

    Raises:
        TypeError: Raise the exception if the type of the labelsis unknown

    Returns:
        pd.DataFrame: The dataframe with the new label column added
    """
    dataframe = data.copy()
    if not labels:
        label_names = pd.Series(np.arange(2))
    elif (isinstance(labels, List) or isinstance(labels, tuple)) and len(labels) == 2:
        label_names = pd.Series(labels)
    else:
        raise TypeError(
            f"The type of the labels : ({labels}) is unknown (should have a int or list)")
    if not label_col_name:
        label_col_name = f"{column}_label"

    dataframe[label_col_name] = label_names.loc[
        pd.Series(
            dataframe[column] >= sep_value if include_sep else dataframe[column] > sep_value,
            dtype=int)].values
    return dataframe


def labelize(
    data: pd.DataFrame,
    column: str,
    labels: Union[int, List, Tuple],
    bounds: Tuple[float, float] = None,
    label_col_name: str = None,
    *args, **kwargs
) -> pd.DataFrame:
    """
    Add a {label_col_name} column to a dataframe containing the labels of a given feature.
    The labels refer to a membership of a values to an interval. The number of intervals is
    determined by the number of labels. All interval have the same size.

    Example:
        Having the features X defined in the interval -1 and 1,
        if we decide to label this features in 4 categories:
        - Values between -1 and -0.5 belong to label 0,
        - The values between -0.5 and 0 belong to the label 1,
        - Values between 0 and 0.5 belong to label 2,
        - Values between 0.5 and 1 belong to label 3.
        0, 1, 2, 3 are labeled features of X.

    If we give a bounds, the intervals are determined inside the bounds.
    Every value which is under or above the bounds will have respectively the label
    '-inf' and '+inf'.

    Args:
        data (pd.DataFrame): The dataframe

        column (str): The column to labelize

        labels (Union[int, List]): A list or tuple of N elements (the labels names). A int to
            determine the number of labels.

        bounds (Tuple[float, float], optional): The intervals are determined inside the bounds.
            Every value which is under or above the bounds will have respectively the label
            '-inf' and '+inf'. Defaults to None.

        label_col_name (str, optional): The column name of the new label column. By default, the
            name is : '{column}_label'. Defaults to None.

    Typical usage example:
        .. code-block:: python

            from mlcf.datatools.utils import labelize

            # A good practice is to take the mean and the standard deviation of the value you want
            # to labelize
            mean = data["return"].mean()
            std = data["return"].std()

            # Here you give the value you want to labelize with column='return'. The new of the
            # labels column will be the name give to 'label_col_name'
            data = labelize(
                data,
                column="return",
                labels=5,
                bounds=(mean-std, mean+std),
                label_col_name="label"
            )

    Raises:
        TypeError: Raise this exception if {bounds} is not a tuple of two elements.

        TypeError: Raise this exception if the type is wrong. Must be a Integer or a List.

        ValueError: If the value of label doesn't have a sense.

    Returns:
        pd.DataFrame: The data frame with the labeled feature added.
    """
    if data.isnull().values.any():
        raise Exception("NaN values has been found")
    dataframe = data.copy()
    if not (isinstance(bounds, tuple) and len(bounds) == 2) and bounds:
        raise TypeError("Bounds must be a tuple of two elements")
    if isinstance(labels, int):
        label_names = pd.Series(np.arange(labels), index=np.arange(labels)+1)
        n_labels = copy(labels)
    elif isinstance(labels, list) or isinstance(labels, tuple):
        label_names = pd.Series(labels, index=np.arange(len(labels))+1)
        n_labels = len(labels)
    else:
        raise TypeError(
            f"The type of the labels : ({labels}) is unknown (should have a int or list)")

    if not label_col_name:
        label_col_name = f"{column}_label"

    if n_labels == 2 and not bounds:
        if isinstance(labels, int):
            return binarize(dataframe, column, *args, **kwargs)
        else:
            return binarize(dataframe, column, labels, *args, **kwargs)

    elif n_labels == 1:
        dataframe[label_col_name] = label_names[1].values

    elif n_labels < 1:
        raise ValueError("A value less than one doen't have sense.")

    else:  # n_labels >= 3

        if bounds:
            intervals = np.linspace(bounds[0], bounds[1], n_labels, endpoint=False)
            label_names[0] = "-inf"
            label_names[n_labels+1] = "+inf"
        else:
            intervals = np.linspace(
                dataframe[column].min(),
                dataframe[column].max(),
                n_labels,
                endpoint=False)

    bins = [-np.inf, *list(intervals)]
    if bounds:
        bins.append(bounds[1])
    bins.append(np.inf)

    if not label_col_name:
        label_col_name = f"{column}_label"
    dataframe[label_col_name] = label_names.loc[
        pd.cut(dataframe[column], bins, right=False).cat.codes].values
    return dataframe


def split_pandas(
    data: pd.DataFrame,
    prop_snd_elem: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe in two dataframes which keep the same columns.
    The {prop_snd_elem} is the proportion of row for the second element.

    Args:
        dataframe (pd.DataFrame): The dataframe we want to split in two

        prop_snd_elem (float, optional): The proportion of row of the second elements in percentage.
            Defaults to 0.5.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The first and second part of the split
    """
    if not isinstance(prop_snd_elem, float):
        raise TypeError("prop_sn_elem must be a float value between 0 and 1")
    dataframe = data.copy()
    if dataframe.empty:
        return (
            pd.DataFrame(columns=dataframe.columns).rename_axis(dataframe.index.name),
            pd.DataFrame(columns=dataframe.columns).rename_axis(dataframe.index.name)
        )
    if prop_snd_elem == 0.0:
        return dataframe, pd.DataFrame(columns=dataframe.columns).rename_axis(dataframe.index.name)
    elif prop_snd_elem == 1.0:
        return pd.DataFrame(columns=dataframe.columns).rename_axis(dataframe.index.name), dataframe
    elif prop_snd_elem < 0.0 or prop_snd_elem > 1.0:
        raise ValueError("prop_sn_elem must be between 0 and 1")
    else:
        times = sorted(dataframe.index)
        second_part = times[-int(prop_snd_elem*len(times))]
        second_data = dataframe[(dataframe.index >= second_part)]
        first_data = dataframe[(dataframe.index < second_part)]
        return first_data, second_data


def split_train_val_test(
    data: pd.DataFrame,
    prop_val_test: float,
    prop_val: float = 0.0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    It splits the dataframe in train, val, and test set.
    The {prop_val_test} value defines the proportion of val and test rows.
    So the proportion of train rows: 1-{prop_val_test}.
    The {prop_val} value defines the proportion of val rows amoung the test set.
    So the proportion of val rows: {prop_val_test}*{prop_val}.
    Finally, the proportion of test rows : {prop_val_test}*(1-{prop_val}).

    Args:
        data (pd.DataFrame): The dataframe we want to split in 3 set: (train, val, test)

        prop_val_test (float): The val and test rows proportion. The proportion of train is
            equal to: 1-{prop_val_test}.

        prop_val (float, optional): The val set proportion amoung the test set. The proportion of
            val rows is : {prop_val_test}*{prop_val}. The proportion of test part is :
            {prop_val_test}*(1-{prop_val}). Defaults to 0.0.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Return a tuple of respectively
        the train dataframe, the val dataframe and the test dataframe.
    """
    dataframe = data.copy()
    train_data, test_val_data = split_pandas(dataframe, prop_snd_elem=prop_val_test)
    val_data, test_data = split_pandas(test_val_data, prop_snd_elem=1-prop_val)
    return train_data, val_data, test_data
