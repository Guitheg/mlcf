
from functools import partial
import pytest
import numpy as np
from mlcf.datatools.sliding_windows import (
    data_windowing,
    predicate_balance_tag,
    predicate_windows_step
)
from mlcf.datatools.standardize_fct import ClassicStd
from numpy.lib.stride_tricks import sliding_window_view


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ({"window_width": 20, "window_step": 1}, {"length": 15328}),
        ({"window_width": 300, "window_step": 1}, {"length": 15048}),
        ({"window_width": 300, "window_step": 2}, {"length": 7524}),
        (
            {
                "window_width": 300,
                "window_step": 2,
                "selected_columns": ["close", "return"]
            },
            {
                "length": 7524
            }
        ),
        (
            {
                "window_width": 300,
                "window_step": 2,
                "selected_columns": ["close", "return"],
                "std_by_feature": {"close": ClassicStd()}
            },
            {
                "length": 7524
            }
        ),
        (
            {
                "window_width": 300,
                "window_step": 2,
                "selected_columns": ["close", "return"],
                "std_by_feature": {"close": ClassicStd()},
                "predicate_row_selection": partial(predicate_windows_step, step_tag_name="step_tag")
            },
            {
                "length": 7524
            }
        ),
        (
            {
                "window_width": 300,
                "window_step": 2,
                "selected_columns": ["close", "return"],
                "std_by_feature": {"close": ClassicStd()},
                "predicate_row_selection": partial(
                    predicate_balance_tag,
                    step_tag_name="step_tag",
                    balance_tag_name="balance_tag")
            },
            {
                "length": 2400
            }
        )
    ]
)
def test_data_windowing(get_btc_tagged_data, test_input, expected):
    data = get_btc_tagged_data(test_input["window_step"])

    list_windows = data_windowing(data, **test_input)
    assert len(list_windows) == expected["length"]
    if "selected_columns" in test_input:
        assert list(list_windows[0].columns) == test_input["selected_columns"]
    if "std_by_feature" in test_input:
        for feature in test_input["std_by_feature"]:
            assert np.round(list_windows[0][feature].mean(), 4) == 0.0
            assert np.round(list_windows[1][feature].std(), 2) == 1.0


@pytest.mark.parametrize(
    "predicate, test_input, expected",
    [
        (
            partial(
                predicate_balance_tag,
                step_tag_name="step_tag",
                balance_tag_name="balance_tag"),
            {"window_step": 1, "window_width": 30},
            2400
        ),
        (
            partial(
                predicate_balance_tag,
                step_tag_name="step_tag",
                balance_tag_name="balance_tag"),
            {"window_step": 2, "window_width": 30},
            2400
        ),
        (
            partial(
                predicate_balance_tag,
                step_tag_name="step_tag",
                balance_tag_name="balance_tag"),
            {"window_step": 20, "window_width": 30},
            766
        ),
        (
            partial(predicate_windows_step, step_tag_name="step_tag"),
            {"window_step": 1, "window_width": 30},
            15318
        ),
        (
            partial(predicate_windows_step, step_tag_name="step_tag"),
            {"window_step": 2, "window_width": 30},
            7659
        ),
        (
            partial(predicate_windows_step, step_tag_name="step_tag"),
            {"window_step": 100, "window_width": 30},
            153
        )
    ]
)
def test_predicate_balance_tag(get_btc_tagged_data, predicate, test_input, expected):
    # Data preprocess --
    data = get_btc_tagged_data(test_input["window_step"])
    data["__index"] = np.arange(len(data))
    windowed_data: np.ndarray = sliding_window_view(
        data,
        window_shape=(test_input["window_width"], len(data.columns))
    )
    windowed_data_shape = (-1, test_input["window_width"], len(data.columns))
    windowed_data = np.reshape(windowed_data, newshape=windowed_data_shape)
    index_data = windowed_data[:, :, list(data.columns).index("__index")]
    # --------------
    bool_list = [
        predicate(data, idx)
        for idx in index_data
    ]
    assert len([b for b in bool_list if b]) == expected
