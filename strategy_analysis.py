from pathlib import Path
from freqtrade.configuration import Configuration
from freqtrade.data.history import load_pair_history
from freqtrade.resolvers import StrategyResolver
import argparse
import os
import sys

def main():
    u = 1000
    avg_profit_per_day = 1.01
    for i in range(2):
        for i in range(364):
            u = u * avg_profit_per_day
    print(u)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-s', "--strategy", default="SampleStrategy", type=str, help="Specify strategy class name which will be used by the bot.")
    # args = parser.parse_args()

    # config = Configuration.from_files([])
    # config["timeframe"] = "5m"
    # config["strategy"] = args.strategy


    # data_location = Path(config['user_data_dir'], 'data', 'binance')
    # if not os.path.isdir(data_location):
    #     raise Exception(f"{data_location} doesn't exist")
    # pair = "BTC/BUSD"
  
  
    # candles = load_pair_history(datadir=data_location,
    #                             timeframe=config["timeframe"],
    #                             pair=pair,
    #                             data_format = "hdf5",
    #                             )   
    # print("Loaded " + str(len(candles)) + f" rows of data for {pair} from {data_location}")
    # candles.head()
    
    # strategy = StrategyResolver.load_strategy(config)
    # df = strategy.analyze_ticker(candles, {'pair': pair})
    # df.tail()
    # print(df)
if __name__ == "__main__":
    main()