# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
from pandas_ta.overlap.ichimoku import ichimoku

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
# import pandas_ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# This class is a sample. Feel free to customize it.
class IchimokuStrategy(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Optimal timeframe for the strategy.
    timeframe = '5m'
    timeframe_mins = timeframe_to_minutes(timeframe)

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.1
    }


    use_custom_stoploss = True
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.2

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.05
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Hyperoptable parameters
    # buy_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    # sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'}
        },
        'subplots': {
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # Ichimoku Kinkō Hyō (ichimoku)
        ichimoku, ichimoku_forward = dataframe.ta.ichimoku(lookahead=False)
        dataframe["ich_tenkan"] = ichimoku["ITS_9"]
        dataframe["ich_kijun"] = ichimoku["IKS_26"]
        dataframe["ich_spanA"] = ichimoku["ISA_9"]
        dataframe["ich_spanB"] = ichimoku["ISB_26"]
        dataframe["ich_chikou"] = dataframe["close"].shift(26)

        # Warning : The use of shift(-26) is only to simulate the leading span A and B.
        # 26 candles forward
        # I duplicate spanA and spanB, shift it by -26 and add at the end leading span a and leading span b.
        # So these data will be available in backtest and in dry/live run
        
        dataframe["ich_lead_spanA"] = pd.concat([ichimoku["ISA_9"], ichimoku_forward["ISA_9"]]).shift(-26)
        dataframe["ich_lead_spanB"] = pd.concat([ichimoku["ISB_26"], ichimoku_forward["ISB_26"]]).shift(-26)
        
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic, returning the new distance relative to current_rate (as ratio).
        e.g. returning -0.05 would create a stoploss 5% below current_rate.
        The custom stoploss can never be below self.stoploss, which serves as a hard maximum loss.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns the initial stoploss value
        Only called when use_custom_stoploss is set to True.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New stoploss value, relative to the current rate
        """
        
        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        stoploss_price = last_candle["ich_spanA"]
        
        if stoploss_price < current_rate:
            return (stoploss_price / current_rate) - 1
         
        return 1

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        
        # Indicators used
        # dataframe["ich_tenkan"]
        # dataframe["ich_kijun"]
        # dataframe["ich_spanA"]
        # dataframe["ich_spanB"]
        # dataframe["ich_chikou"]
        # dataframe["ich_lead_spanA"]
        # dataframe["ich_lead_spanB"]

        dataframe.loc[
        (
            (dataframe["ich_lead_spanA"] < dataframe["ich_lead_spanB"]) &
            (dataframe["ich_spanB"] < dataframe["ich_spanA"]) &
            (dataframe['close'] > dataframe["ich_spanB"]) &  
            (dataframe["ich_chikou"] > dataframe["ich_spanB"]) &
            (dataframe["ich_tenkan"] > dataframe["ich_kijun"]) &
            (dataframe['volume'] > 0)  # Make sure Volume is not 0
        ),
        ['buy', 'buy_tag']] = (1, 'buy_signal_ichimoku')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        dataframe.loc[
        (
            (dataframe["ich_lead_spanA"] < dataframe["ich_lead_spanB"]) &
            (dataframe["ich_spanB"] > dataframe["ich_spanA"]) &
            (dataframe['close'] < dataframe["ich_spanA"]) &  
            (dataframe["ich_chikou"] < dataframe["ich_spanA"]) &
            (dataframe["ich_tenkan"] < dataframe["ich_kijun"])
        ),
        ['sell', 'exit_tag']] = (1, 'sell_signal_ichimoku')
        return dataframe
