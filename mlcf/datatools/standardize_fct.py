"""_summary_
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

__all__ = [
    "StandardisationFct",
    "ClassicStd",
    "MinMaxStd",
    "standardize",
    "standardize_fit_transform"
]


class IdentityTransform():
    def transform(self, x: pd.Series):
        return x.loc[:]


class StandardisationFct():
    def __init__(self, class_std=IdentityTransform, *args, **kwargs):
        self.class_std = class_std
        self.args = args
        self.kwargs = kwargs
        self.std = self.class_std(*self.args, **self.kwargs)

    def partial_fit(self, data: pd.Series):
        pass

    def fit(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]):
        pass

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def transform(
        self,
        data: Union[np.ndarray, pd.DataFrame, pd.Series]
    ) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        if isinstance(data, pd.Series):
            series = data.copy()
            series.loc[:] = self.std.transform(np.reshape(series.values, (-1, 1))).reshape(-1)
            return series
        elif isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray):
            return self.std.transform(data)

    def copy(self):
        return self.__class__(*self.args, **self.kwargs)


class ClassicStd(StandardisationFct):
    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True
    ):
        super(ClassicStd, self).__init__(StandardScaler, with_mean=with_mean, with_std=with_std)

    def partial_fit(self, data: pd.Series):
        self.std.partial_fit(np.reshape(data.values, (-1, 1)))

    def fit(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]):
        if isinstance(data, pd.Series):
            self.std.fit(np.reshape(data.values, (-1, 1)))
        else:
            self.std.fit(data)


class MinMaxStd(StandardisationFct):
    def __init__(
        self,
        minmax: Tuple[float, float] = (0, 1),
        feature_range: Tuple[float, float] = (0, 1)
    ):
        super(MinMaxStd, self).__init__(MinMaxScaler, feature_range=feature_range)
        self.minmax = minmax

    def partial_fit(self, data: pd.Series):
        if self.minmax:
            self.std.partial_fit([[self.minmax[0]], [self.minmax[1]]])
        else:
            self.std.partial_fit(np.reshape(data.values, (-1, 1)))

    def fit(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]):
        if self.minmax:
            if isinstance(data, pd.Series):
                self.std.fit([[self.minmax[0]], [self.minmax[1]]])
            else:
                self.std.fit(np.array([[self.minmax[0], self.minmax[1]]]*data.shape[-1]).T)
        else:
            if isinstance(data, pd.Series):
                self.std.fit(np.reshape(data.values, (-1, 1)))
            else:
                self.std.fit(data)


def copy_std_feature_dict(std_by_feature: Dict[str, StandardisationFct]):
    return {
        feature: std_obj.copy() for feature, std_obj in std_by_feature.items()
    }


# TODO: (enhancement) implement option for inplace = False
# TODO: (refactoring) list comprehension for data_transformed or verify if it's inplace or not
def standardize(
    fit_data: Union[pd.DataFrame, List[pd.DataFrame]],
    transform_data: Union[pd.DataFrame, List[pd.DataFrame]],
    std_by_feature: Optional[Dict[str, StandardisationFct]] = None,
    inplace: bool = True,
    std_fct_save: bool = True
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    if not std_fct_save and std_by_feature is not None:
        std_by_feature = copy_std_feature_dict(std_by_feature)
    if not inplace:
        raise NotImplementedError
    if std_by_feature is None:
        return transform_data
    if isinstance(fit_data, pd.DataFrame):
        if not fit_data.empty:
            for feature, std_obj in std_by_feature.items():
                std_obj.partial_fit(fit_data[feature])
    elif (isinstance(fit_data, list) and
          np.all([isinstance(i, pd.DataFrame) for i in fit_data])):
        for data in fit_data:
            if not data.empty:
                for feature, std_obj in std_by_feature.items():
                    std_obj.partial_fit(data[feature])
    else:
        raise TypeError(f"Bad fit_data ({type(fit_data)}) type in standardize")

    if isinstance(transform_data, pd.DataFrame):
        if not transform_data.empty:
            for feature, std_obj in std_by_feature.items():
                transform_data[feature] = std_obj.transform(transform_data[feature])
    elif (isinstance(transform_data, list) and
          np.all([isinstance(i, pd.DataFrame) for i in transform_data])):
        for data in transform_data:
            if not data.empty:
                for feature, std_obj in std_by_feature.items():
                    data[feature] = std_obj.transform(data[feature])
    else:
        raise TypeError(f"Bad transform_data ({type(transform_data)}) type in standardize")

    return transform_data


def standardize_fit_transform(
    fit_transform_data: Union[pd.DataFrame, List[pd.DataFrame]],
    *args, **kwargs
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    return standardize(fit_transform_data, fit_transform_data, *args, **kwargs)
