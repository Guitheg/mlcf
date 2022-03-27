# MLCF - Machine Learning Toolkit for Cryptocurrency Forecasting  

This library provides tools for cryptocurrency forecasting and trade decision making.  
For now, the library provide only data tools, such as:

- OHLCV file reader
- Tools to add extern and intern indicators
- Tools to labelize data
- Tools to work on a set of intervals in order
- Tools to windowed the data
- Tools to build, save and read a dataset
- Tools to standardize data
- Tools to preprocess the data by filtering some windows

This library doesn't provide models or an end-to-end trade bot.

For more information, find the documentation here : https://guitheg.github.io/mlcf/

---

## Installation

OS officially supported:  

- **Linux**  

Python version officially supported:  

- **3.7**  

- **3.8**  

- **3.9**

To succeed the installation, it needs to install some dependencies, which are:

- the TA-LIB C library

---

### Installation for Linux (python v3.7, v3.8, v3.9)

- TA-LIB C library installation:  

*Note: the talib-install.sh file and the ta-lib-0.4.0-src.tar.gz archive will be downloaded on your PC. They can be manually deleted at the end of the installation.*  

```bash
wget https://raw.githubusercontent.com/Guitheg/mlcf/main/build_helper/talib-install.sh
sh talib-install.sh
```

- MLCF package

```bash
pip install mlcf
```

---

## MLCF example module usage

In this part, we will introduce some example usage of MLCF module.

---

### File reader module

```python
# -----------  read file ---------------------------------
from pathlib import Path
from mlcf.datatools.data_reader import (
    read_ohlcv_json_from_file,
    read_ohlcv_json_from_dir,
    read_json_file
)

# from a ohlcv json file
data = read_ohlcv_json_from_file(Path("tests/testdata/ETH_BUSD-15m.json"))

# from a directory, a pair, and a timeframe
pair = "ETH_BUSD"
tf = "15m"
data = read_ohlcv_json_from_dir(Path("tests/testdata/"), pair=pair, timeframe=tf)

# read a json file (but not necessary a OHLCV file)
data = read_json_file(Path("tests/testdata/meteo.json"), 'time', ["time", "Temperature"])

# -------------------------------------------------------
```

### Indicator Module

```python
# ------------------- Indicators module -----------------------------
from mlcf.indicators.add_indicators import add_intern_indicator

# you can add yoursel your own indicators or features
data["return"] = data["close"].pct_change(1)
data.dropna(inplace=True)  # make sure to drop nan values

# you can add intern indicator
data = add_intern_indicator(data, indice_name="adx")
# -------------------------------------------------------
```

### Label Tool

```python
# ------------------- Labelize Tool -----------------------------
from mlcf.datatools.utils import labelize

# A good practice is to take the mean and the standard deviation of the value you want to
# labelize
mean = data["return"].mean()
std = data["return"].std()

# Here you give the value you want to labelize with column='return'. The new of the labels column
# will be the name give to 'label_col_name'
data = labelize(
    data,
    column="return",
    labels=5,
    bounds=(mean-std, mean+std),
    label_col_name="label"
)
```

### Data Intervals Module, Standardization Tools and WindowFilter Tool

```python
# ------------------- Data Intervals Module and Standardization Tools -----------------------------
from mlcf.datatools.data_intervals import DataIntervals
from mlcf.datatools.standardize_fct import ClassicStd, MinMaxStd
from mlcf.datatools.windowing.filter import LabelBalanceFilter

# We define a dict which give us the information about what standardization apply to each columns.
std_by_feautures = {
    "close": ClassicStd(),
    "return": ClassicStd(with_mean=False),  # to avoid to shift we don't center
    "adx": MinMaxStd(minmax=(0, 100))  # the value observed in the adx are between 0 and 100 and we
                                       # want to set it between 0 and 1.
}
data_intervals = DataIntervals(data, n_intervals=10)
data_intervals.standardize(std_by_feautures)

# We can apply a filter the dataset we want. Here we will filter the values in order to balance
# the histogram of return value. For this, we use the label previously process on return.
filter_by_set = {
    "train": LabelBalanceFilter("label")  # the column we will balance the data is 'label
                                          # the max count will be automatically process
}

# dict_train_val_test is a dict with the key 'train', 'val', 'test'. The value of the dict is a
# WTSeries (a windowed time series).
dict_train_val_test = data_intervals.windowing(
    window_width=30,
    window_step=1,
    selected_columns=["close", "return", "adx"],
    filter_by_dataset=filter_by_set,
    std_by_feature=None  # Here we can pass the same kind of dict previously introduce to apply
                         # the standardization independtly on each window
)
# -------------------------------------------------------
```

### Window Iterator Tool

```python
# -------------------- Window Iterator Tool --------------------

# If we don't want to use the Data Interval Module. We can simple use a WTSeries with our data.
from mlcf.datatools.windowing.tseries import WTSeries

# To create a WTSeries from pandas.DataFrame
wtseries = WTSeries.create_wtseries(
    dataframe=data,
    window_width=30,
    window_step=1,
    selected_columns=["close", "return", "adx"],
    window_filter=LabelBalanceFilter("label"),
    std_by_feature=std_by_feautures
)

# Or from a wtseries .h5 file:
wtseries = WTSeries.read(Path("/tests/testdata/wtseries.h5"))

# We can save the wtseries as a file.
wtseries.write(Path("/tests/testdata", "wtseries"))

# we can iterate over the wtseries:
for window in wtseries:
    pass
    # Where window is a pd.Dataframe representing a window.

# -------------------------------------------------------
```

### Forecast Window Iterator Tool

```python
# -------------------- Forecast Window Iterator Tool --------------------

# This class allow us to iterate over a WTSeries but the iteration
# (__getitem__) give us a tuple of 2

from mlcf.datatools.windowing.forecast_iterator import WindowForecastIterator

data_train = WindowForecastIterator(
    wtseries,
    input_width=29,
    target_width=1,  # The sum of the input_width and target_width must not exceed the window width
                     # of the wtseries
    input_features=["close", "adx"],
    target_features=["return"]
)
for window in data_train:
    window_input, window_target = window
    pass
# -------------------------------------------------------
```
