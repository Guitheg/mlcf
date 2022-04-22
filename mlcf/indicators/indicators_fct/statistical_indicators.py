"""TSFRESH Features Module

See https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html for more info.
"""

from functools import partial
import pandas as pd
from typing import Callable, Dict
from tsfresh.feature_extraction import feature_calculators as fc

__all__ = [
    "TSFRESH_FEATURES",
    "add_tsfresh_feature"
]

TSFRESH_FEATURES: Dict[str, Callable] = {
    "abs_energy": fc.abs_energy,
    "absolute_maximum": fc.absolute_maximum,
    "absolute_sum_of_changes": fc.absolute_sum_of_changes,
    "agg_autocorrelation": fc.agg_autocorrelation,
    "agg_linear_trend": fc.agg_linear_trend,
    "approximate_entropy": fc.approximate_entropy,
    "ar_coefficient": fc.ar_coefficient,
    "augmented_dickey_fuller": fc.augmented_dickey_fuller,
    "autocorrelation": fc.autocorrelation,
    "benford_correlation": fc.benford_correlation,
    "binned_entropy": fc.binned_entropy,
    "c3": fc.c3,
    "change_quantiles": fc.change_quantiles,
    "cid_ce": fc.cid_ce,
    "count_above": fc.count_above,
    "count_above_mean": fc.count_above_mean,
    "count_below": fc.count_below,
    "count_below_mean": fc.count_below_mean,
    "cwt_coefficients": fc.cwt_coefficients,
    "energy_ratio_by_chunks": fc.energy_ratio_by_chunks,
    "fft_aggregated": fc.fft_aggregated,
    "fft_coefficient": fc.fft_coefficient,
    "first_location_of_maximum": fc.first_location_of_maximum,
    "first_location_of_minimum": fc.first_location_of_minimum,
    "fourier_entropy": fc.fourier_entropy,
    "friedrich_coefficients": fc.friedrich_coefficients,
    "has_duplicate": fc.has_duplicate,
    "has_duplicate_max": fc.has_duplicate_max,
    "has_duplicate_min": fc.has_duplicate_min,
    "index_mass_quantile": fc.index_mass_quantile,
    "kurtosis": fc.kurtosis,
    "large_standard_deviation": fc.large_standard_deviation,
    "last_location_of_maximum": fc.last_location_of_maximum,
    "last_location_of_minimum": fc.last_location_of_minimum,
    "lempel_ziv_complexity": fc.lempel_ziv_complexity,
    "length": fc.length,
    "linear_trend": fc.linear_trend,
    "linear_trend_timewise": fc.linear_trend_timewise,
    "longest_strike_above_mean": fc.longest_strike_above_mean,
    "longest_strike_below_mean": fc.longest_strike_below_mean,
    "matrix_profile": fc.matrix_profile,
    "max_langevin_fixed_point": fc.max_langevin_fixed_point,
    "maximum": fc.maximum,
    "mean": fc.mean,
    "mean_abs_change": fc.mean_abs_change,
    "mean_change": fc.mean_change,
    "mean_n_absolute_max": fc.mean_n_absolute_max,
    "mean_second_derivative_central": fc.mean_second_derivative_central,
    "median": fc.median,
    "minimum": fc.minimum,
    "number_crossing_m": fc.number_crossing_m,
    "number_cwt_peaks": fc.number_cwt_peaks,
    "number_peaks": fc.number_peaks,
    "partial_autocorrelation": fc.partial_autocorrelation,
    "percentage_of_reoccurring_datapoints_to_all_datapoints":
        fc.percentage_of_reoccurring_datapoints_to_all_datapoints,
    "percentage_of_reoccurring_values_to_all_values":
        fc.percentage_of_reoccurring_values_to_all_values,
    "permutation_entropy": fc.permutation_entropy,
    "quantile": fc.quantile,
    "query_similarity_count": fc.query_similarity_count,
    "range_count": fc.range_count,
    "ratio_beyond_r_sigma": fc.ratio_beyond_r_sigma,
    "ratio_value_number_to_time_series_length": fc.ratio_value_number_to_time_series_length,
    "root_mean_square": fc.root_mean_square,
    "sample_entropy": fc.sample_entropy,
    "set_property": fc.set_property,
    "skewness": fc.skewness,
    "spkt_welch_density": fc.spkt_welch_density,
    "standard_deviation": fc.standard_deviation,
    "sum_of_reoccurring_data_points": fc.sum_of_reoccurring_data_points,
    "sum_of_reoccurring_values": fc.sum_of_reoccurring_values,
    "sum_values": fc.sum_values,
    "symmetry_looking": fc.symmetry_looking,
    "time_reversal_asymmetry_statistic": fc.time_reversal_asymmetry_statistic,
    "value_count": fc.value_count,
    "variance": fc.variance,
    "variance_larger_than_standard_deviation": fc.variance_larger_than_standard_deviation,
    "variation_coefficient": fc.variation_coefficient
}


def add_tsfresh_feature(
    data: pd.DataFrame,
    tsfresh_feature_name: str,
    column: str,
    timeperiod: int,
    custom_column_name: str = None,
    *args, **kwargs
) -> pd.DataFrame:
    """Add a tsfresh (statistical) feature to the dataframe.

    See https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html for more info.

    Args:
        data (pd.DataFrame): The dataframe.

        tsfresh_feature_name (str): the name of the added feature. Please choose among this list:
            {feature_list}.

        column (str): The column name to which apply the calculations from.

        timeperiod (int): The timeperiod of the rolling window.

        custom_column_name (str, optional): The new name of the added column. Defaults to None.

    Returns:
        pd.DataFrame: The dataframe with the added feature.
    """
    dataframe = data.copy()
    if custom_column_name is None:
        param_string: str = "" if not len(kwargs) else f"{kwargs}"
        custom_column_name = f"{tsfresh_feature_name}({column})[{timeperiod}]{param_string}"

    partial_function = partial(TSFRESH_FEATURES[tsfresh_feature_name], *args, **kwargs)
    dataframe[custom_column_name] = dataframe[column].rolling(timeperiod).apply(partial_function)
    return dataframe


add_tsfresh_feature.__doc__ = str(add_tsfresh_feature.__doc__).format(
    feature_list=list(TSFRESH_FEATURES.keys())
)
