
from functools import partial
from pathlib import Path
import pandas
import pytest
import numpy as np
from mlcf.datatools.data_intervals import DataInIntervals
from mlcf.datatools.windowing.tseries import (
    WTSeries,
    predicate_balance_tag,
    predicate_windows_step
)
from mlcf.datatools.standardize_fct import ClassicStd
from numpy.lib.stride_tricks import sliding_window_view


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ({"window_width": 20, "window_step": 1},
         {"length": 15328, "first_window": lambda data: data["return"].iloc[0:20].values}),
        ({"window_width": 30, "window_step": 1},
         {"length": 15318, "first_window": lambda data: data["return"].iloc[0:30].values}),
        ({"window_width": 30, "window_step": 2},
         {"length": 7659, "first_window": lambda data: data["return"].iloc[0:30].values}),
        (
            {
                "window_width": 30,
                "window_step": 2,
                "selected_columns": ["close", "return"]
            },
            {
                "length": 7659,
                "first_window": lambda data: data["return"].iloc[0:30].values
            }
        ),
        (
            {
                "window_width": 30,
                "window_step": 2,
                "selected_columns": ["close", "return"],
                "std_by_feature": {"close": ClassicStd()}
            },
            {
                "length": 7659,
                "first_window": lambda data: data["return"].iloc[0:30].values
            }
        ),
        (
            {
                "window_width": 30,
                "window_step": 2,
                "selected_columns": ["close", "return"],
                "std_by_feature": {"close": ClassicStd()},
                "predicate_row_selection": partial(predicate_windows_step, step_tag_name="step_tag")
            },
            {
                "length": 7659,
                "first_window": lambda data: data["return"].iloc[1:31].values
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
                "length": 2400,
                "first_window": lambda data: data["return"].iloc[959:1259].values
            }
        )
    ]
)
def test_wtseries(get_btc_tagged_data, test_input, expected):
    data = get_btc_tagged_data(test_input["window_step"])

    wtseries = WTSeries.create_wtseries(data, **test_input)
    assert len(wtseries) == expected["length"]
    assert np.all(wtseries[0]["return"].values == expected["first_window"](data))
    if "selected_columns" in test_input:
        assert list(wtseries.features) == test_input["selected_columns"]
        assert wtseries.ndim == len(test_input["selected_columns"])
    else:
        assert wtseries.ndim == len(list(data.columns))
    if "std_by_feature" in test_input:
        for feature in test_input["std_by_feature"]:
            assert np.round(wtseries[0][feature].mean(), 4) == 0.0
            assert np.round(wtseries[0][feature].std(), 1) == 1.0
    assert wtseries.n_window == expected["length"]
    assert wtseries.width == test_input["window_width"]


@pytest.mark.parametrize(
    "test_input",
    [
        ({"window_width": 20, "window_step": 1}),
        ({"window_width": 30, "window_step": 10}),
        (
            {
                "window_width": 30,
                "window_step": 2,
                "selected_columns": ["close", "return"]
            }
        ),
        (
            {
                "window_width": 30,
                "window_step": 2,
                "selected_columns": ["close", "return"],
                "std_by_feature": {"close": ClassicStd()}
            }
        )
    ]
)
def test_merge(ohlcvra_btc, test_input):
    data_intervals = DataInIntervals.create_list_interval(ohlcvra_btc, n_intervals=2)
    wtseries_1 = WTSeries.create_wtseries(data_intervals[0], **test_input)
    wtseries_2 = WTSeries.create_wtseries(data_intervals[0], **test_input)
    wtseries = wtseries_1.merge(wtseries_2)
    assert np.all(wtseries[len(wtseries_1)].values == wtseries_2[0].values)
    assert len(wtseries_1) + len(wtseries_2) == len(wtseries)


@pytest.mark.parametrize(
    "test_input, group_key",
    [
        ({"window_width": 20, "window_step": 1}, None),
        ({"window_width": 30, "window_step": 10}, None),
        (
            {
                "window_width": 30,
                "window_step": 2,
                "selected_columns": ["close", "return"]
            },
            "train"
        ),
    ]
)
def test_write(ohlcvr_btc, test_input, group_key, tmp_path: Path):
    wtseries = WTSeries.create_wtseries(ohlcvr_btc, **test_input)
    userdir = tmp_path.joinpath("userdir")
    userdir.mkdir()
    file_name = "datasetwts"
    file_path = wtseries.write(userdir, file_name, group_key)
    assert file_path.is_file()
    assert np.all(pandas.read_hdf(file_path).values == wtseries.data.values)


@pytest.mark.parametrize(
    "test_input, group_key",
    [
        ({"window_width": 20, "window_step": 1}, None),
        ({"window_width": 30, "window_step": 10}, None),
        (
            {
                "window_width": 30,
                "window_step": 2,
                "selected_columns": ["close", "return"]
            },
            "train"
        ),
    ]
)
def test_read(ohlcvr_btc, test_input, group_key, tmp_path: Path):
    wtseries = WTSeries.create_wtseries(ohlcvr_btc, **test_input)
    userdir = tmp_path.joinpath("userdir")
    userdir.mkdir()
    file_name = "datasetwts"
    file_path = wtseries.write(userdir, file_name, group_key)
    assert file_path.is_file()
    wtseries_read = WTSeries.read(file_path, group_key)
    assert np.all(wtseries.data.values == wtseries_read.data.values)


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
def test_predicate_tag(get_btc_tagged_data, predicate, test_input, expected):
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
