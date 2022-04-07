from functools import partial
import pandas as pd
import numpy as np
from tsfresh.feature_extraction import feature_calculators as fc


def add_mean(data: pd.DataFrame, column: str, timeperiod: int, new_column_name: str = None):
    dataframe = data.copy()
    if not new_column_name:
        new_column_name = f"{column}[mean({timeperiod})]"
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).mean()


def add_std(data: pd.DataFrame, column: str, timeperiod: int, new_column_name: str = None):
    dataframe = data.copy()
    if not new_column_name:
        new_column_name = f"{column}[std({timeperiod})]"
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).std()


def add_var(data: pd.DataFrame, column: str, timeperiod: int, new_column_name: str = None):
    dataframe = data.copy()
    if not new_column_name:
        new_column_name = f"{column}[var({timeperiod})]"
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).var()


def add_quantile(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    quantile: float = 0.5,
    new_column_name: str = None
):

    dataframe = data.copy()
    if not new_column_name:
        new_column_name = f"{column}[quantile_{quantile}({timeperiod})]"
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).quantile(quantile=quantile)


def add_min(data: pd.DataFrame, column: str, timeperiod: int, new_column_name: str = None):
    dataframe = data.copy()
    if not new_column_name:
        new_column_name = f"{column}[min({timeperiod})]"
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).min()


def add_max(data: pd.DataFrame, column: str, timeperiod: int, new_column_name: str = None):
    dataframe = data.copy()
    if not new_column_name:
        new_column_name = f"{column}[max({timeperiod})]"
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).max()


def add_kurtosis(data: pd.DataFrame, column: str, timeperiod: int, new_column_name: str = None):
    dataframe = data.copy()
    if not new_column_name:
        new_column_name = f"{column}[kurtosis({timeperiod})]"
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).kurt()


def add_skewness(data: pd.DataFrame, column: str, timeperiod: int, new_column_name: str = None):
    dataframe = data.copy()
    if not new_column_name:
        new_column_name = f"{column}[skewness({timeperiod})]"
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).skew()


def add_corr(
    data: pd.DataFrame,
    column: str,
    other_column: str,
    timeperiod: int,
    new_column_name: str = None
):
    dataframe = data.copy()
    if not new_column_name:
        new_column_name = f"{column}[correlation({timeperiod})]"
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).corr(dataframe[other_column])


def add_cov(
    data: pd.DataFrame,
    column: str,
    other_column: str,
    timeperiod: int,
    new_column_name: str = None
):
    dataframe = data.copy()
    if not new_column_name:
        new_column_name = f"{column}[covariance({timeperiod})]"
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).cov(dataframe[other_column])


def add_absolute(data: pd.DataFrame, column: str, new_column_name: str = None):
    dataframe = data.copy()
    if not new_column_name:
        new_column_name = f"{column}[absolute]"
    dataframe[new_column_name] = dataframe[column].abs()


def add_sum(data: pd.DataFrame, column: str, timeperiod: int, new_column_name: str = None):
    dataframe = data.copy()
    if not new_column_name:
        new_column_name = f"{column}[sum({timeperiod})]"
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).sum()


def add_pct_change(data: pd.DataFrame, column: str, shift: int = 1, new_column_name: str = None):
    dataframe = data.copy()
    if not new_column_name:
        new_column_name = f"{column}[pct_change(t-{shift})]"
    dataframe[new_column_name] = dataframe[column].pct_change(shift)


def add_agg_autocorrelation(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    maxlag: int = 1,
    new_column_name: str = None
):
    dataframe = data.copy()
    if not new_column_name:
        new_column_name = f"{column}[agg_autocorr({timeperiod})(n={maxlag})]"

    agg_autocorr = partial(fc.agg_autocorrelation, maxlag=maxlag)
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).apply(agg_autocorr)


def add_agg_linear_trend(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    new_column_name: str = None
):
    raise NotImplementedError


def add_approximate_entropy(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    m: int,
    r: float,
    new_column_name: str = None
):
    dataframe = data.copy()

    if not new_column_name:
        new_column_name = f"{column}[approximate_entropy({timeperiod})(m={m}|r={r})]"

    approximate_entropy = partial(fc.approximate_entropy, m=m, r=r)
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).apply(approximate_entropy)


def add_ar_coefficient(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    new_column_name: str = None
):
    raise NotImplementedError


def add_augmented_dickey_fuller(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    new_column_name: str = None
):
    raise NotImplementedError


def add_autocorrelation(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    lag: int,
    new_column_name: str = None
):
    dataframe = data.copy()

    if not new_column_name:
        new_column_name = f"{column}[autocorrelation({timeperiod})(lag={lag})]"

    autocorrelation = partial(fc.autocorrelation, lag=lag)
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).apply(autocorrelation)


def add_benford_correlation(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    new_column_name: str = None
):
    dataframe = data.copy()

    if not new_column_name:
        new_column_name = f"{column}[benford_correlation({timeperiod})]"

    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).apply(fc.benford_correlation)


def add_binned_entropy(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    max_bins: int,
    new_column_name: str = None
):
    dataframe = data.copy()

    if not new_column_name:
        new_column_name = f"{column}[benford_correlation({timeperiod})(maxbins={max_bins})]"

    binned_entropy = partial(fc.binned_entropy, max_bins=max_bins)
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).apply(binned_entropy)


def add_c3(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    lag: int = 1,
    new_column_name: str = None
):
    dataframe = data.copy()

    if not new_column_name:
        new_column_name = f"{column}[c3({timeperiod})(lag={lag})]"

    c3 = partial(fc.c3, lag=lag)
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).apply(c3)


def add_change_quantiles(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    ql: float,
    qh: float,
    isabs: bool,
    f_agg: str = "mean",
    new_column_name: str = None
):
    dataframe = data.copy()

    if not new_column_name:
        new_column_name = (
            f"{column}[change_quantiles({timeperiod})" +
            "(ql={ql}|qh={qh}|isabs={isabs}|fagg={f_agg})]")

    change_quantiles = partial(fc.change_quantiles, ql=ql, qh=qh, isabs=isabs, f_agg=f_agg)
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).apply(change_quantiles)


def add_cid_ce(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    normalize: bool,
    new_column_name: str = None
):
    raise NotImplementedError


def add_count_above(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    treshold: bool,
    new_column_name: str = None
):
    dataframe = data.copy()

    if not new_column_name:
        new_column_name = f"{column}[count_above({timeperiod})(treshold={treshold})]"

    count_above = partial(fc.count_above, t=treshold)
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).apply(count_above)


def add_count_above_mean(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    treshold: bool,
    new_column_name: str = None
):
    dataframe = data.copy()

    if not new_column_name:
        new_column_name = f"{column}[count_above_mean({timeperiod})]"

    count_above_mean = partial(fc.count_above_mean, t=treshold)
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).apply(count_above_mean)


def add_count_below(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    treshold: bool,
    new_column_name: str = None
):
    dataframe = data.copy()

    if not new_column_name:
        new_column_name = f"{column}[count_below({timeperiod})(treshold={treshold})]"

    count_below = partial(fc.count_below, t=treshold)
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).apply(count_below)


def add_count_below_mean(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    treshold: bool,
    new_column_name: str = None
):
    dataframe = data.copy()

    if not new_column_name:
        new_column_name = f"{column}[count_below_mean({timeperiod})]"

    count_below_mean = partial(fc.count_below_mean, t=treshold)
    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).apply(count_below_mean)


def add_cwt_coefficients(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    width: np.ndarray,
    coeff: int,
    w: int,
    new_column_name: str = None
):
    raise NotImplementedError


def add_energy_ratio_by_chunks(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    width: np.ndarray,
    coeff: int,
    w: int,
    new_column_name: str = None
):
    raise NotImplementedError


def add_fft_aggregated(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    aggtype: str,
    new_column_name: str = None
):
    raise NotImplementedError


def add_fft_coefficient(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    attr: str,
    coeff: int,
    new_column_name: str = None
):
    raise NotImplementedError


def add_first_location_of_maximum(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    new_column_name: str = None
):
    dataframe = data.copy()

    if not new_column_name:
        new_column_name = f"{column}[first_location_of_maximum({timeperiod})]"

    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).apply(
        fc.first_location_of_maximum)


def add_first_location_of_minimum(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    new_column_name: str = None
):
    dataframe = data.copy()

    if not new_column_name:
        new_column_name = f"{column}[first_location_of_minimum({timeperiod})]"

    dataframe[new_column_name] = dataframe[column].rolling(timeperiod).apply(
        fc.first_location_of_minimum)


def add_fourier_entropy(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    new_column_name: str = None
):
    raise NotImplementedError


def add_friedrich_coefficients(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    m: int,
    r: int,
    coeff: int,
    new_column_name: str = None
):
    raise NotImplementedError


def add_has_duplicate(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    new_column_name: str = None
):
    raise NotImplementedError


def add_has_duplicate_max(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    new_column_name: str = None
):
    raise NotImplementedError


def add_has_duplicate_min(
    data: pd.DataFrame,
    column: str,
    timeperiod: int,
    new_column_name: str = None
):
    raise NotImplementedError
