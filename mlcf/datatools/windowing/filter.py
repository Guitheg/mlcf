
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


# TODO: (enhancement) use only __call__
class WindowFilter(ABC):
    def __init__(self):
        self.data: pd.DataFrame = None

    @abstractmethod
    def __call__(
        self,
        data: pd.DataFrame,
        index_array: np.ndarray,
        *args, **kwargs
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> bool:
        pass


class LabelBalanceFilter(WindowFilter):
    def __init__(
        self,
        column: str,
        max_count: Optional[int] = None,
        sample_function: Callable = random.sample
    ):

        super(LabelBalanceFilter, self).__init__()
        self.column = column
        self.tag_name = "label_balance_tag"
        self.max_count: Optional[int] = max_count
        self.sample_function = sample_function

    def __call__(
        self,
        data: pd.DataFrame,
        index_array: np.ndarray,
        *args, **kwargs
    ):
        """
        This function tags False the values that are not taken into account in the construction
        of the histogram of the relevant {column} values of {data}.
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

    def __getitem__(self, idx: List[int]) -> bool:
        return self.data.loc[self.data.index[idx[-1]], self.tag_name]
