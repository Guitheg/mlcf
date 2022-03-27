"""Standardisation Function Module.

This module provide the StandardisationModule class which provide methods compatible with
:py:class:`DataIntevals <mlcf.datatools.data_intervals.DataIntervals>`
and :py:class:`WTSeries <mlcf.windowing.iterator.tseries.WTSeries>` to perform standardisation
on features.
"""

from __future__ import annotations
from abc import ABC
from typing import Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# TODO (doc) review

__all__ = [
    "StandardisationModule",
    "ClassicStd",
    "MinMaxStd",
    "standardize",
    "standardize_fit_transform"
]


class StandardisationModule(ABC):
    """The Standardisation Module abstract class.

    This abstract class is used as a basis for implementing interfaces to sklearn's standardisation
    classes. The interest of having these interfaces allows to be compatible with pandas.DataFrame
    and to reshape the data.

    Attributes:

        class_std (Callable): A class which is used for the standardisation.
            It must implements following methods : fit(), partial_fit(), transform().

        args (list): All parameters that will be passed to class_std()

        kwargs (dict): All parameters that will be passed to class_std()

        std (object): The object used for the standardisation.
            It has been constructed from class_std(\\*args, \\*\\*kwargs).

    """

    def __init__(self, class_std: Callable, *args, **kwargs):
        self.class_std = class_std
        self.args = args
        self.kwargs = kwargs
        self.std = self.class_std(*self.args, **self.kwargs)

    def partial_fit(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """If this function is not implemented, then any fit is applied.

        Args:
            data (Union[np.ndarray, pd.DataFrame, pd.Series]): The data.
        """

        pass

    def fit(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """If this function is not implemented, then any fit is applied.

        Args:
            data (Union[np.ndarray, pd.DataFrame, pd.Series]): The data.
        """

        pass

    def fit_transform(
        self,
        data: Union[np.ndarray, pd.DataFrame, pd.Series]
    ) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """It runs :py:func:`~fit` then it returns the result of :py:func:`~transform`.

        Args:
            data (Union[np.ndarray, pd.DataFrame, pd.Series]): The data.

        Returns:
            Union[np.ndarray, pd.DataFrame, pd.Series]: The standardised data.
        """
        self.fit(data)
        return self.transform(data)

    def transform(
        self,
        data: Union[np.ndarray, pd.DataFrame, pd.Series]
    ) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """It returns the result of self.std.transform(data). The transform method standardise
        the data.

        Args:
            data (Union[np.ndarray, pd.DataFrame, pd.Series]): The data.

        Raises:
            TypeError: If the data isn't neither a pandas.DataFrame,
                a pandas.Series or a numpy.ndarray.

        Returns:
            Union[np.ndarray, pd.DataFrame, pd.Series]: The standardised data.
        """
        if isinstance(data, pd.Series):
            series = data.copy()
            series.iloc[:] = self.std.transform(np.reshape(data.values, (-1, 1))).reshape(-1)
            return series
        elif isinstance(data, pd.DataFrame):
            dataframe = data.copy()
            dataframe.iloc[:] = self.std.transform(data.values)
            return dataframe
        elif isinstance(data, np.ndarray):
            return self.std.transform(data)
        else:
            raise TypeError("data must be a numpy.ndarray, pandas.DataFrame or a pandas.Series")

    def copy(self) -> StandardisationModule:
        """Returns a copy of this object but forgets the adjustment (fit).

        Returns:
            StandardisationModule: An empty copy.
        """
        return StandardisationModule(self.class_std, *self.args, **self.kwargs)


class ClassicStd(StandardisationModule):
    """Allows to perform a standardisation with :py:class:`sklearn.preprocessing.StandardScaler`.
    Standardize features by removing the mean and scaling to unit variance. z = (x - u) / s.

    """

    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True
    ):
        """We can choose if we want to remove the mean and/or scale to unit variance.

        Args:
            with_mean (bool, optional): True if we want to remove the mean. Defaults to True.
            with_std (bool, optional): True if we want to scale to unit variance. Defaults to True.
        """
        super(ClassicStd, self).__init__(StandardScaler, with_mean=with_mean, with_std=with_std)

    def partial_fit(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]):
        if isinstance(data, pd.Series):
            self.std = self.std.partial_fit(np.reshape(data.values, (-1, 1)))
        elif isinstance(data, pd.DataFrame):
            self.std = self.std.partial_fit(data.values)
        elif isinstance(data, np.ndarray):
            self.std = self.std.partial_fit(data)
        else:
            raise TypeError("data must be a numpy.ndarray, pandas.DataFrame or a pandas.Series")

    def fit(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]):
        if isinstance(data, pd.Series):
            self.std = self.std.fit(np.reshape(data.values, (-1, 1)))
        elif isinstance(data, pd.DataFrame):
            self.std = self.std.fit(data.values)
        elif isinstance(data, np.ndarray):
            self.std = self.std.fit(data)
        else:
            raise TypeError("data must be a numpy.ndarray, pandas.DataFrame or a pandas.Series")


class MinMaxStd(StandardisationModule):
    """Allows to perform a standardisation with :py:class:`sklearn.preprocessing.MinMaxScaler`.
    Transform features by scaling each feature to a given range.
    """
    def __init__(
        self,
        minmax: Tuple[float, float] = (0, 1),
        feature_range: Tuple[float, float] = (0, 1)
    ):
        """The features will be scale in the range given by {feature_range}.
        Instead of using the fit and partial fit methods, we can set the min and max values with
        the {minmax} parameter.

        Args:
            minmax (Tuple[float, float], optional): It allows to avoid the fit and partial_fit
                methods by setting the minimum and maximum value. Defaults to (0, 1).
            feature_range (Tuple[float, float], optional):
                The future range over which features will be scaled. Defaults to (0, 1).
        """
        super(MinMaxStd, self).__init__(MinMaxScaler, feature_range=feature_range)
        self.minmax = minmax

    def partial_fit(self, data: pd.Series):
        if self.minmax:
            if isinstance(data, pd.Series):
                self.std = self.std.partial_fit([[self.minmax[0]], [self.minmax[1]]])
            elif isinstance(data, pd.DataFrame):
                self.std = self.std.partial_fit(
                    np.array([[self.minmax[0], self.minmax[1]]]*data.values.shape[-1]).T)
            elif isinstance(data, np.ndarray):
                self.std = self.std.partial_fit(
                    np.array([[self.minmax[0], self.minmax[1]]]*data.shape[-1]).T)
            else:
                raise TypeError("data must be a numpy.ndarray, pandas.DataFrame or a pandas.Series")
        else:
            if isinstance(data, pd.Series):
                self.std = self.std.partial_fit(np.reshape(data.values, (-1, 1)))
            elif isinstance(data, pd.DataFrame):
                self.std = self.std.partial_fit(data.values)
            elif isinstance(data, np.ndarray):
                self.std = self.std.partial_fit(data)
            else:
                raise TypeError("data must be a numpy.ndarray, pandas.DataFrame or a pandas.Series")

    def fit(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]):
        if self.minmax:
            if isinstance(data, pd.Series):
                self.std = self.std.fit([[self.minmax[0]], [self.minmax[1]]])
            elif isinstance(data, pd.DataFrame):
                self.std = self.std.fit(
                    np.array([[self.minmax[0], self.minmax[1]]]*data.values.shape[-1]).T)
            elif isinstance(data, np.ndarray):
                self.std = self.std.fit(
                    np.array([[self.minmax[0], self.minmax[1]]]*data.shape[-1]).T)
            else:
                raise TypeError("data must be a numpy.ndarray, pandas.DataFrame or a pandas.Series")
        else:
            if isinstance(data, pd.Series):
                self.std = self.std.fit(np.reshape(data.values, (-1, 1)))
            elif isinstance(data, pd.DataFrame):
                self.std = self.std.fit(data.values)
            elif isinstance(data, np.ndarray):
                self.std = self.std.fit(data)
            else:
                raise TypeError("data must be a numpy.ndarray, pandas.DataFrame or a pandas.Series")


def copy_std_feature_dict(
    std_by_feature: Dict[str, StandardisationModule]
) -> Dict[str, StandardisationModule]:
    """It copy each StandardisationModule of a dictionnary in a new dictionnary.

    Args:
        std_by_feature (Dict[str, StandardisationModule]): A dictionnary with the format:
            {key (string) -> :py:class:`StandardisationModule
            <mlcf.datatools.standardisation.StandardisationModule>`}.

    Returns:
        Dict[str, StandardisationModule]: The copied dictionnary.
    """
    return {
        feature: std_obj.copy() for feature, std_obj in std_by_feature.items()
    }


# TODO (enhancement) implement option for inplace = False /change not implemented in doc
# TODO (refactoring) list comprehension for data_transformed or verify if it's inplace or not
def standardize(
    fit_data: Union[pd.DataFrame, List[pd.DataFrame]],
    transform_data: Union[pd.DataFrame, List[pd.DataFrame]],
    std_by_feature: Optional[Dict[str, StandardisationModule]] = None,
    inplace: bool = True,
    std_fct_save: bool = True
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """It allows you to fit from one list of data frames and then standardize on another
    list of data frames.

    (Not Implemented: inplace operation)

    Args:
        fit_data (Union[pd.DataFrame, List[pd.DataFrame]]): The data frame or list of data frame
            used to fit the StandardisationModule.

        transform_data (Union[pd.DataFrame, List[pd.DataFrame]]): The data frame or list of data
            frame which will be standardise.

        std_by_feature (Optional[Dict[str, StandardisationModule]], optional): A dictionary
            prodiving the standardisation to be applied on each column.
            The dictionary format must be as following:
            {string -> :py:class:`StandardisationModule
            <mlcf.datatools.standardisation.StandardisationModule>`}.
            The key must correspond to a column name (a feature) of the data frame.
            The value is any object inheriting from the
            :py:class:`StandardisationModule
            <mlcf.datatools.standardisation.StandardisationModule>` class.
            Defaults to None.

        inplace (bool, optional): True the {transform_data} is modify inplace.
            Otherwise it returns a copy of the modified {transform_data} (not implemented).
            Defaults to True.

        std_fct_save (bool, optional):Set to True so that the adjustment is in place,
        otherwise set to False. In any case, the std_by_feature is not returned. Defaults to True.

    Raises:
        NotImplementedError: _description_
        TypeError: _description_
        TypeError: _description_

    Returns:
        Union[pd.DataFrame, List[pd.DataFrame]]: _description_
    """
    if std_by_feature is None:
        return transform_data
    else:
        if not std_fct_save:
            dict_std_by_feature: Dict[str, StandardisationModule] = \
                copy_std_feature_dict(std_by_feature)
        else:
            dict_std_by_feature = std_by_feature

    if not inplace:
        raise NotImplementedError

    if isinstance(fit_data, pd.DataFrame):
        if not fit_data.empty:
            for feature, std_obj in dict_std_by_feature.items():
                std_obj.fit(fit_data[feature])

    elif (isinstance(fit_data, list) and np.all([isinstance(i, pd.DataFrame) for i in fit_data])):
        for data in fit_data:
            if not data.empty:
                for feature, std_obj in dict_std_by_feature.items():
                    std_obj.partial_fit(data[feature])

    else:
        raise TypeError(f"Bad fit_data ({type(fit_data)}) type in standardize")

    if isinstance(transform_data, pd.DataFrame):
        if not transform_data.empty:
            for feature, std_obj in dict_std_by_feature.items():
                transform_data[feature] = std_obj.transform(transform_data[feature])

    elif (isinstance(transform_data, list) and
          np.all([isinstance(i, pd.DataFrame) for i in transform_data])):
        for data in transform_data:
            if not data.empty:
                for feature, std_obj in dict_std_by_feature.items():
                    data[feature] = std_obj.transform(data[feature])
    else:
        raise TypeError(f"Bad transform_data ({type(transform_data)}) type in standardize")

    return transform_data


def standardize_fit_transform(
    fit_transform_data: Union[pd.DataFrame, List[pd.DataFrame]],
    *args, **kwargs
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """It perform a fit transform using :py:class:`~standardize`.

    Args:
        fit_transform_data (Union[pd.DataFrame, List[pd.DataFrame]]):
            The data used to fit and which will be transformed.

    Returns:
        Union[pd.DataFrame, List[pd.DataFrame]]: The transformed data.
    """
    return standardize(fit_transform_data, fit_transform_data, *args, **kwargs)
