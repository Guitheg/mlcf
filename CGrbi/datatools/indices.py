from typing import List
from functools import partial
import pandas as pd

# def add_indicator(indice_name : str, dataframe : pd.DataFrame):
#     data = dataframe.copy()
#     match indice_name:
#         case "ADX": dataframe['adx'] = ta.ADX(dataframe)


def add_indicators(dataframe : pd.DataFrame, list_indices : List[str], dropna : bool = True):
    data = dataframe.copy()
    
    # for indice in list_indices:
    #     data = add_indicator(indice, data)
    
    data.dropna(inplace=dropna)
    return data

