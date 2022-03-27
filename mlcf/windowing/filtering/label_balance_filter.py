import random
from typing import Callable, List, Optional
import numpy as np
import pandas as pd

from mlcf.windowing.filtering import WindowFilter


# TODO (doc) review
class LabelBalanceFilter(WindowFilter):
    """Label balancing filter.

    It will even out the histogram of a labeled feature.

    A labeled feature corresponds to the category of a value.
    Example: Having the feature X defined in the interval -1 and 1,
    if we decide to label this feature in 4 categories:
    - Values between -1 and -0.5 belong to label 0,
    - The values between -0.5 and 0 belong to the label 1,
    - Values between 0 and 0.5 belong to label 2,
    - Values between 0.5 and 1 belong to label 3.
    0, 1, 2, 3 are labeled feature of X.

    The histogram of these labeled features is constructed by taking into account the
    last index of each window.

    From a parameter called maximum occurrence, we select a number of lines so that the occurrence
    of each labeled feature does not exceed said parameter.

Translated with www.DeepL.com/Translator (free version)

    Attributes:
        column (str): The filter will take only the values of this column in account

        max_occ (int): The maximum number of count for each label after the filtering

        sample_function (Callable): A callable function such as fct(list, k) -> list_of_k_values
    """
    def __init__(
        self,
        column: str,
        max_occ: Optional[int] = None,
        sample_function: Callable = random.sample
    ):
        """Create a Label balancing filter object.

        We need to give the column to know which one will be taken into account to equalize
        the histogram.
        We can define the maximum occurrence parameter.
        Otherwise it will be determined automatically.
        We can give a selection function that will determine the selection method of
        the K windows such that K < {max_occ} for each label.

        Args:
            column (str): The label column on which the histogram is processed.

            max_occ (Optional[int], optional): The maximum number of occurence for each labeled
                feature. Defaults to None.

            sample_function (Callable, optional):
                A selection function that respects this format: fct(list, k) -> list_of_k_values.
                Defaults to random.sample.
        """

        super(LabelBalanceFilter, self).__init__()
        self.column = column
        self.max_occ: Optional[int] = max_occ
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
        """Given the time series dataframe and an index array corresponding to the time index in
        each window, it returns a list of booleans indicating whether or not to take the window.

        Args:
            data (pd.DataFrame): The time series dataframe

            index_array (np.ndarray): The index array correspond to the windowed index dataframe.
                It's a 2-D array with first axis give the windows and the second axis gives the
                row index in the time series dataframe.

        Returns:
            List[bool]: A list of boolean indicating whether or not to take a window.
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

        # if {max_occ} has not been set, the {max_occ} is set has the mean between the number
        # of label which belongs to '-inf' and '+inf'
        if not self.max_occ:
            if "-inf" in value_count and "+inf" in value_count:
                max_occ = np.mean([value_count["-inf"], value_count["+inf"]]).astype(int)
            else:
                max_occ = np.mean(value_count).astype(int)
        else:
            max_occ = self.max_occ

        # for each value_count greater than the max count we deselect number of rows equals to
        # the difference between the value_count and the max_occ
        for idx in value_count.index:
            if value_count[idx] > max_occ:
                tag_col.loc[
                    sorted(self.sample_function(
                        list(dataframe[dataframe[self.column] == idx].index),
                        k=value_count[idx]-max_occ)
                    )
                ] = False
        self.data = data.copy()
        self.data[self.tag_name] = tag_col
        return [self.data.loc[self.data.index[idx[-1]], self.tag_name] for idx in index_array]
