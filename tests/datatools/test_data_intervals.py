
import pytest
import pandas as pd
import numpy as np
from mlcf.datatools.data_intervals import DataIntervals
from mlcf.datatools.standardize_fct import ClassicStd, MinMaxStd
from mlcf.datatools.windowing.filter import LabelBalanceFilter


@pytest.mark.parametrize(
    "data_selection, test_input, expected_exception",
    [
        ("empty", {"n_intervals": 1}, ValueError),
        ("full", {"n_intervals": 1.5}, TypeError),
        ("full", {"n_intervals": 0}, ValueError),
    ]
)
def test_create_list_interval_exception(data_selection, test_input, expected_exception, ohlcv_btc):
    data = {
        "empty": pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]).rename_axis("date"),
        "full": ohlcv_btc
    }
    with pytest.raises(expected_exception):
        DataIntervals.create_list_interval(data=data[data_selection], **test_input)


@pytest.mark.parametrize(
    "test_input, expected_a, expected_b",
    [
        ({"n_intervals": 1}, 15348, 15348),
        ({"n_intervals": 2}, 7674, 7674),
        ({"n_intervals": 1000}, 16, 15),
    ]
)
def test_create_list_interval(test_input, expected_a, expected_b, ohlcv_btc):
    res = DataIntervals.create_list_interval(data=ohlcv_btc, **test_input)
    assert len(res[0]) == expected_a
    assert len(res[-1]) == expected_b
    assert sum([len(i) for i in res]) == len(ohlcv_btc)


@pytest.mark.parametrize(
    "n_intervals, test_input, train_set, val_set, test_set",
    [
        (10, {"prop_val_test": 0.5, "prop_val": 0.5}, 768, 384, 383),
        (10, {"prop_val_test": 0.0, "prop_val": 0.0}, 1535, 0, 0),
        (10, {"prop_val_test": 1.0, "prop_val": 0.0}, 0, 0, 1535),
        (10, {"prop_val_test": 1.0, "prop_val": 1.0}, 0, 1535, 0),
    ]
)
def test_train_val_test(test_input, n_intervals, train_set, val_set, test_set, ohlcv_btc):
    list_intervals = DataIntervals.create_list_interval(data=ohlcv_btc, n_intervals=n_intervals)
    splited = DataIntervals.split_train_val_test(list_intervals=list_intervals, **test_input)
    assert len(splited[0]) == n_intervals
    assert len(splited[1]) == n_intervals
    assert len(splited[2]) == n_intervals
    assert len(splited[0][0]) == train_set
    assert len(splited[1][0]) == val_set
    assert len(splited[2][0]) == test_set
    lengths_array = np.array([[len(data) for data in row] for i, row in enumerate(splited)])
    assert np.sum(lengths_array) == len(ohlcv_btc)
    assert np.all(
        np.sum(lengths_array, axis=0) == np.array([[len(data) for data in list_intervals]]))


@pytest.mark.parametrize(
    "test_input, train_set, val_set, test_set",
    [
        ({"n_intervals": 5, "prop_val_test": 0.5, "prop_val": 0.5}, 1535, 768, 767),
    ]
)
def test_data_intervals(ohlcv_btc, test_input, train_set, val_set, test_set):
    data_in_intervals = DataIntervals(ohlcv_btc, **test_input)
    assert len(data_in_intervals.intervals["train"][0]) == train_set
    assert len(data_in_intervals.intervals["val"][0]) == val_set
    assert len(data_in_intervals.intervals["test"][0]) == test_set
    assert data_in_intervals.n_intervals == len(data_in_intervals.intervals["train"])
    assert data_in_intervals.n_intervals == test_input["n_intervals"]


@pytest.mark.parametrize(
    "test_input",
    [
        (
            {
                "n_intervals": 5,
                "prop_val_test": 0.5,
                "prop_val": 0.5
            }
        )
    ]
)
def test_standardize(ohlcvra_btc, test_input):
    std_by_feature = {
        "close": ClassicStd(),
        "return": ClassicStd(with_mean=False),
        "adx": MinMaxStd(minmax=(0, 100))
    }
    data_intervals = DataIntervals(ohlcvra_btc, **test_input)
    data_intervals.standardize(std_by_feature)

    data = pd.DataFrame()
    for interval in data_intervals["train"]:
        data = pd.concat([data, interval])
    assert np.round(data["close"].std(), 3) == 1.0
    assert np.round(data["close"].mean(), 3) == 0.0
    assert np.round(data["return"].std(), 3) == 1.0
    assert np.round(data["adx"].min(), 3) >= 0.0
    assert np.round(data["adx"].max(), 3) <= 1.0


@pytest.mark.parametrize(
    "n_intervals, test_input, expected",
    [
        (
            4,
            {
                "window_width": 100,
                "window_step": 1,
                "selected_columns": ["close", "return"],
                "std_by_feature": {
                    "close": ClassicStd(),
                    "return": ClassicStd(with_mean=False)
                }
            },
            11883
        ),
        (
            4,
            {
                "window_width": 10,
                "window_step": 1,
                "selected_columns": ["close", "return"],
                "std_by_feature": {
                    "close": ClassicStd(),
                    "return": ClassicStd(with_mean=False)
                }
            },
            12243
        ),
        (
            4,
            {
                "window_width": 100,
                "window_step": 3,
                "selected_columns": ["close", "return"],
                "std_by_feature": {
                    "close": ClassicStd(),
                    "return": ClassicStd(with_mean=False)
                },
                "filter_by_dataset":
                    {
                        "train": LabelBalanceFilter("label", 200)
                    }
            },
            3765
        ),
        (
            4,
            {
                "window_width": 100,
                "window_step": 3,
                "selected_columns": ["close", "return"],
                "std_by_feature": {
                    "close": ClassicStd(),
                    "return": ClassicStd(with_mean=False)
                },
                "filter_by_dataset":
                    {
                        "train": LabelBalanceFilter("label", 200)
                    }
            },
            3765
        ),
    ]
)
def test_data_windowing(ohlcvrl_btc, n_intervals, test_input, expected):
    data = ohlcvrl_btc
    data_intervals = DataIntervals(data, n_intervals=n_intervals)
    dataset = data_intervals.data_windowing(**test_input)

    assert len(dataset["train"]) == expected

    for key in dataset:
        np.all(
            dataset[key][0].index.values ==
            data_intervals.intervals[key][0]
            .iloc[:test_input["window_width"]][test_input["selected_columns"]].index.values)
