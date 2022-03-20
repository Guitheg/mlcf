"""_summary_
"""

from typing import Callable, Dict, List, Optional, Union, Tuple
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import numpy as np

from mlcf.datatools.standardize_fct import StandardisationFct, standardize_fit_transform


__all__ = [
    "predicate_windows_step",
    "predicate_balance_tag",
    "data_windowing"
]


def predicate_windows_step(
    data,
    window_width,
    window_step,
    idx,
    step_tag_name: str
) -> bool:
    return data.loc[data.index[(idx*window_step)+window_width-1], step_tag_name]


def predicate_balance_tag(
    data,
    window_width,
    window_step,
    idx,
    balance_tag_name: str,
    step_tag_name: str
) -> bool:
    return (
        data.loc[data.index[(idx*window_step)+window_width-1], balance_tag_name] and
        predicate_windows_step(data, window_width, window_step, idx, step_tag_name))


# TODO: (opti) parrallelize ?
# TODO: Verify window step if it's not déphasé
def data_windowing(
    dataframe: pd.DataFrame,
    window_width: int,
    window_step: int,
    selected_columns: Optional[List[str]] = None,
    predicate_row_selection: Optional[Callable] = None,
    std_by_feature: Optional[Dict[str, StandardisationFct]] = None
) -> List[pd.DataFrame]:

    data = dataframe.copy()
    if len(data) == 0 or len(data) < window_width:
        return [pd.DataFrame(columns=data.columns)]
    n_windows = ((len(data.index)-window_width) // window_step) + 1
    n_columns = len(data.columns)

    # Slid window on all data
    windowed_data: np.ndarray = sliding_window_view(
        data,
        window_shape=(window_width, len(data.columns))
    )

    # Reshape windowed data
    windowed_data_shape: Tuple[int, int, int] = (n_windows, window_width, n_columns)
    windowed_data = np.reshape(windowed_data, newshape=windowed_data_shape)

    # Make list of window (dataframe)
    list_windows: List[pd.DataFrame] = [
        standardize_fit_transform(
            std_by_feature,
            pd.DataFrame(
                window,
                index=data.index[idx*window_step: (idx*window_step)+window_width],
                columns=data.columns)[selected_columns if selected_columns else data.columns],
            )
        for idx, window in enumerate(windowed_data)
        if not predicate_row_selection or predicate_row_selection(
            data,
            window_width,
            window_step,
            idx)
    ]

    return list_windows
