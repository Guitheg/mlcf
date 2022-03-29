from pathlib import Path
from pickle import NONE

from numpy import float64
from mlcf.datatools.data_reader import (
    read_ohlcv_json_from_file
)
from mlcf.datatools.data_intervals import DataIntervals
from mlcf.datatools.standardize_fct import ClassicStd, MinMaxStd
from mlcf.datatools.windowing.filter import LabelBalanceFilter
from mlcf.datatools.windowing.forecast_iterator import WindowForecastIterator
from mlcf.datatools.windowing.tseries import WTSeries
from mlcf.datatools.indicators.add_indicators import add_intern_indicator
from mlcf.datatools.indicators.indicators_tools import bollinger_bands, rsi, sma, mid_price
from mlcf.datatools.utils import labelize
import pandas as pd


# from a ohlcv json file
path = Path('/home/embraysite/Documents/GitHub/mlcf/mlcf_home/data/binance/1INCH_BUSD-1h.json')
data = read_ohlcv_json_from_file(path)

# -- Indicator Module
# ------------------- Indicators module -----------------------------

data.dropna(inplace=True)
# you can add yoursel your own indicators or features
data["return1"] = data["close"].pct_change(1)
data["return5"] = data["close"].pct_change(5)
data["return10"] = data["close"].pct_change(10)
data["return20"] = data["close"].pct_change(20)
bb_aux = bollinger_bands(data["close"], 20, 2, "bb20-2")
data = pd.concat([data, bb_aux], axis=1)
bb_aux = bollinger_bands(data["close"], 30, 2, "bb30-2")
data = pd.concat([data, bb_aux], axis=1)
bb_aux = bollinger_bands(data["close"], 30, 2, "bb20-2.5")
data = pd.concat([data, bb_aux], axis=1)
data["rsi14"] = rsi(data["close"], 14)
data["rsi10"] = rsi(data["close"], 10)
data["rsi20"] = rsi(data["close"], 20)
data["sma10"] = sma(mid_price(data), 10)
data["sma20"] = sma(mid_price(data), 20)
data["sma30"] = sma(mid_price(data), 30)
data.dropna(inplace=True)  # make sure to drop nan values
# you can add intern indicator
data = add_intern_indicator(data, indice_name="adx")
data.dropna(inplace=True)
# -------------------------------------------------------

# -- Label Tool

# ------------------- Labelize Tool -----------------------------

# A good practice is to take the mean and the standard deviation of the value you want to
# labelize
label_list = ["return1", "return5", "return10", "return20"]
i = 0
for name in label_list:
    i += 1
    mean = data[name].mean()
    std = data[name].std()

    # Here you give the value you want to labelize with column='return'.
    # The new of the labels column will be the name give to 'label_col_name'
    data = labelize(
        data,
        column=name,
        labels=2,
        label_col_name="label"+str(i)
    )

# -- Data Intervals Module, Standardization Tools and WindowFilter Tool

# ------------------- Data Intervals Module and Standardization Tools -----------------------------


# We define a dict which give us the information about what standardization apply to each columns.
std_by_features = {
    "return1": ClassicStd(with_mean=False),  # to avoid to shift we do no center the mean
    "return5": ClassicStd(with_mean=False),
    "return10": ClassicStd(with_mean=False),
    "return20": ClassicStd(with_mean=False),
    "adx": MinMaxStd(minmax=(0, 100)),  # the value observed in the adx are between 0 and 100 and we
    "rsi14": MinMaxStd(minmax=(0, 100)),  # want to set it between 0 and 1.
    "rsi10": MinMaxStd(minmax=(0, 100)),
    "rsi20": MinMaxStd(minmax=(0, 100))
}
data_intervals = DataIntervals(data, n_intervals=10)
data_intervals.standardize(std_by_features)

# We can apply a filter the dataset we want. Here we will filter the values in order to balance
# the histogram of return value. For this, we use the label previously process on return.
filter_by_set = {
    "train": LabelBalanceFilter("return1_label")  # the column we will balance the data is 'label
                                          # the max count will be automatically process
}

# dict_train_val_test is a dict with the key 'train', 'val', 'test'. The value of the dict is a
# WTSeries (a windowed time series).
window_std_by_features = {
    "close": ClassicStd(),
    "bb20-2upper": ClassicStd(),
    "bb20-2mid": ClassicStd(),
    "bb20-2lower": ClassicStd(),
    "bb30-2upper": ClassicStd(),
    "bb30-2mid": ClassicStd(),
    "bb30-2lower": ClassicStd(),
    "bb20-2.5upper": ClassicStd(),
    "bb20-2.5mid": ClassicStd(),
    "bb20-2.5lower": ClassicStd(),
    "sma10": ClassicStd(),
    "sma20": ClassicStd(),
    "sma30": ClassicStd()
}

print(data.columns)

dict_train_val_test = data_intervals.data_windowing(
    window_width=5,
    window_step=1,
    selected_columns=None,  # None is all collumns :o
    filter_by_dataset=filter_by_set,
    std_by_feature=window_std_by_features  # Here we can pass the same kind of dict previously
    # introduce to apply the standardization independtly on each window
)
# -------------------------------------------------------

# -- Window Iterator Tool

# -------------------- Window Iterator Tool --------------------

# If we don't want to use the Data Interval Module. We can simple use a WTSeries with our data.

# To create a WTSeries from pandas.DataFrame
# wtseries = WTSeries.create_wtseries(
#     dataframe=data,
#     window_filter=30,
#     window_step=1,
#     selected_columns=["close", "return1", "adx"],
#     std_by_feature=window_std_by_features
# )

# # Or from a wtseries .h5 file:
# wtseries = WTSeries.read(Path("/tests/testdata/wtseries.h5"))

# # We can save the wtseries as a file.
# wtseries.write(Path("/tests/testdata", "wtseries"))

# # we can iterate over the wtseries:
# for window in wtseries:
#     pass
#     # Where window is a pd.Dataframe representing a window.

# -------------------------------------------------------

# -- Forecast Window Iterator Tool

# -------------------- Forecast Window Iterator Tool --------------------

# This class allow us to iterate over a WTSeries but the iteration
# (__getitem__) give us a tuple of 2


data_train = WindowForecastIterator(
    dict_train_val_test['train'],
    input_width=4,
    target_width=1,  # The sum of the input_width and target_width must not exceed the window width
                     # of the wtseries
    input_features=["close", "adx", "return1", "rsi14"],
    target_features=["return5"]
)
j = 0
# print(data[["close", "adx", "return1", "rsi14", "return5"]])
# for window in data_train:
#     window_input, window_target = window
#     if j == 0:
#         print("window_input")
#         print(window_input)
#         print("window_target")
#         print(window_target[0][0])
#         print("_-------------------------_----------___")
#         print(data.loc(data["return5"].astype(float64) == window_target[0][0]))
#         j += 1
