
from cgi import test
import pytest
import pandas as pd
import numpy as np
from mlcf.datatools.data_intervals import AnyStepTag, DataInIntervals, HaveAlreadyAStepTag, LabelBalanceTag, TagCreator
from mlcf.datatools.standardize_fct import ClassicStd, MinMaxStd
from mlcf.datatools.utils import labelize


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
        DataInIntervals.create_list_interval(data=data[data_selection], **test_input)


@pytest.mark.parametrize(
    "test_input, expected_a, expected_b",
    [
        ({"n_intervals": 1}, 15348, 15348),
        ({"n_intervals": 2}, 7674, 7674),
        ({"n_intervals": 1000}, 16, 15),
    ]
)
def test_create_list_interval(test_input, expected_a, expected_b, ohlcv_btc):
    res = DataInIntervals.create_list_interval(data=ohlcv_btc, **test_input)
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
    list_intervals = DataInIntervals.create_list_interval(data=ohlcv_btc, n_intervals=n_intervals)
    splited = DataInIntervals.split_train_val_test(list_intervals=list_intervals, **test_input)
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
def test_data_in_intervals(ohlcv_btc, test_input, train_set, val_set, test_set):
    data_in_intervals = DataInIntervals(ohlcv_btc, **test_input)
    assert len(data_in_intervals.intervals["train"][0]) == train_set
    assert len(data_in_intervals.intervals["val"][0]) == val_set
    assert len(data_in_intervals.intervals["test"][0]) == test_set
    assert data_in_intervals.n_intervals == len(data_in_intervals.intervals["train"])
    assert data_in_intervals.n_intervals == test_input["n_intervals"]
    assert not data_in_intervals.step_tag


# Test if the intervals are well standardized.
#
# @pytest.mark.parametrize(
#     "std_by_feature, test_input, mean, std, maxi, mini",
#     [
#         ({}, {"n_intervals": 5}, 0.0, 1.0, 2.0, -2.0),
#         ({"close": ClassicStd()}, {"n_intervals": 5}, 0.0, 1.0, 2.0, -2.0),
#         ({"close": MinMaxStd()}, {"n_intervals": 5}, 0.0, 1.0, 2.0, -2.0),
#         ({"close": MinMaxStd()}, {"n_intervals": 5}, 0.0, 1.0, 2.0, -2.0),
#     ]
# )
# def test_standardize(std_by_feature, test_input, mean, std, maxi, mini, ohlcv_btc):
#     data_intervals = DataInIntervals(ohlcv_btc, **test_input)
#     data_intervals.standardize(std_by_feature)
#     for key in std_by_feature:
#         for set_name, set in data_intervals.intervals.items():
#             for interval in set:


def test_add_step_tag_exception_1(ohlcv_btc):
    data_intervals = DataInIntervals(ohlcv_btc, n_intervals=10)
    data_intervals.add_step_tag(1)
    with pytest.raises(HaveAlreadyAStepTag):
        data_intervals.add_step_tag(1)


def test_add_step_tag_exception_2(ohlcv_btc):
    data_intervals = DataInIntervals(ohlcv_btc, n_intervals=10)
    with pytest.raises(ValueError):
        data_intervals.add_step_tag(0)


@pytest.mark.parametrize(
    "test_step",
    [
        (1),
        (2),
        (100),
    ]
)
def test_add_step_tag(ohlcv_btc, test_step):
    data_intervals = DataInIntervals(ohlcv_btc, n_intervals=10)
    data_intervals.add_step_tag(test_step)
    for _, intervals in data_intervals.intervals.items():
        for interval in intervals:
            assert np.all(interval["step_tag"].iloc[::test_step])
            assert len(interval.loc[~interval["step_tag"].values]) == int(
                (1 - (1/test_step)) * len(interval))


def test_add_tag_exception(ohlcv_btc):
    data_intervals = DataInIntervals(ohlcv_btc, n_intervals=10)
    with pytest.raises(AnyStepTag):
        data_intervals.add_tag(
            LabelBalanceTag(),
            tag_name="balance_tag",
            list_partitions=["train"],
            column="volume")


@pytest.mark.parametrize(
    "test_step, tag_creator, expected_a, expected_b",
    [
        (1, TagCreator(), 1228, 1228),
        (2, TagCreator(), 1228, 614),
        (1, LabelBalanceTag(), 768, 768),
        (2, LabelBalanceTag(), 373, 373),
        (2, LabelBalanceTag(max_count=200), 512, 512)
    ]
)
def test_add_tag(ohlcvrl_btc, test_step, tag_creator, expected_a, expected_b):
    tag_name = "balance_tag"
    data_intervals = DataInIntervals(ohlcvrl_btc, n_intervals=10)
    data_intervals.add_step_tag(test_step)
    data_intervals.add_tag(
        tag_creator,
        tag_name=tag_name,
        list_partitions=["train"],
        column="label"
    )
    df = data_intervals("train")[0]
    assert len(df.loc[df[tag_name]]) == expected_a
    assert len(df.loc[(df[tag_name]) & (df["step_tag"])]) == expected_b


@pytest.mark.parametrize(
    "n_labels, max_count, expected",
    [
        (10, 150, [13547, 1800]),
        (5, 300, [13247, 2100])
    ]
)
def test_label_balance_tag(n_labels, max_count, expected, ohlcvr_btc):
    mean = ohlcvr_btc["return"].mean()
    std = ohlcvr_btc["return"].std()
    ohlcvrl_btc = labelize(
        data=ohlcvr_btc,
        column="return",
        labels=n_labels,
        bounds=(mean - std, mean + std),
        label_col_name="label"
    )
    label_creator = LabelBalanceTag(max_count)
    balanced_label = label_creator(ohlcvrl_btc, "label")
    assert list(balanced_label.value_counts()) == expected
    assert list(ohlcvrl_btc.loc[balanced_label.values, "label"].value_counts()) == \
        [max_count] * (n_labels + 2)
