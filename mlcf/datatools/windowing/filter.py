
from abc import ABC, abstractmethod
import random
from typing import Callable, List, Optional
import numpy as np
import pandas as pd

# TODO: (doc)

__all__ = [
    "WindowFilter",
    "LabelBalanceFilter"
]


class WindowFilter(ABC):
    """
    This class allows us to perform a window filtering according to some condition. This class is compatible with WTSeries and DataIntervals.
    Every class which want to perform window filtering must inherit from this class.
    """

    @abstractmethod
    def __call__(
        self,
        data: pd.DataFrame,
        index_array: np.ndarray,
        *args, **kwargs
    ) -> pd.DataFrame:
        pass


class LabelBalanceFilter(WindowFilter):
    """
    From a set of window, will filter the window in order to uniformize the histogram of the given label column.
    
    Attributes:
        column (str): The filter will take only the values of this column in account
        
        max_count (int): The maximum number of count for each label after the filtering
        
        sample_function (Callable): A callable function such as fct(list, k) -> list_of_k_values
    """
    def __init__(
        self,
        column: str,
        max_count: Optional[int] = None,
        sample_function: Callable = random.sample
    ):
        """
        Initialize the LabelBalanceFilter.
        We must give the column to know which will be take in account to uniformize the histogram.

        Args:
            column (str): The label column on which the histogram is processed.

            max_count (Optional[int], optional): The maximum number of counted label after filtering. Defaults to None.

            sample_function (Callable, optional): A function such as fct(list, k) -> list_of_k_values.
                Defaults to random.sample.
        """

        super(LabelBalanceFilter, self).__init__()
        self.column = column
        self.max_count: Optional[int] = max_count
        self.sample_function: Callable = sample_function

    @property
    def tag_name(self):
        return "label_balance_tag"

    def __call__(
        self,
        data: pd.DataFrame,
        index_array: np.ndarray,
        *args, **kwargs
    ) -> List[bool]:
        """
        Given the time series dataframe and a index array corresponding to the windowed index of the time series dataframe.

        Args:
            data (pd.DataFrame): The time series dataframe

            index_array (np.ndarray): the index array correspond to the windowed index dataframe. It's a 2-D array with first axis give the 
                windows and the second axis gives the row index in the time series dataframe.

        Returns:
            List[bool]: List length = number of window. With True if we keep the corresponding window else False.
        """

        dataframe = data.copy()

        # extract every last index which of each window
        last_index_of_each_window = index_array[:, -1]

        # tag col with all values set to false
        tag_col = pd.Series([False]*len(dataframe), index=dataframe.index)

        # set true every row corresponding to a last index of a window
        tag_col.iloc[last_index_of_each_window] = True

        # filter and keep only row corresponding to a last index of a window
        dataframe = dataframe.iloc[last_index_of_each_window]

        # count the number of label for eache label (the corresponding label is given by the column)
        value_count = dataframe[self.column].value_counts()

        # if {max_count} has not been set, the {max_count} is set has the mean between the number
        # of label which belongs to '-inf' and '+inf'
        if not self.max_count:
            if "-inf" in value_count and "+inf" in value_count:
                max_count = np.mean([value_count["-inf"], value_count["+inf"]]).astype(int)
            else:
                max_count = np.mean(value_count).astype(int)
        else:
            max_count = self.max_count

        # for each value_count greater than the max count we deselect number of rows equals to
        # the difference between the value_count and the max_count
        for idx in value_count.index:
            if value_count[idx] > max_count:
                tag_col.loc[
                    sorted(self.sample_function(
                        list(dataframe[dataframe[self.column] == idx].index),
                        k=value_count[idx]-max_count)
                    )
                ] = False
        self.data = data.copy()
        self.data[self.tag_name] = tag_col
        return [self.data.loc[self.data.index[idx[-1]], self.tag_name] for idx in index_array]
