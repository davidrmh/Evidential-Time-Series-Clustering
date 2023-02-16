import os
import jax.numpy as jnp
import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional
from jaxtyping import Float, Array
from copy import deepcopy


class StockData(dict):
    def __init__(self, path: str, old_first: bool = True, sep: str = "_"):
        """
        Args:
            :param path(str): path of the directory containing the csv file with stock prices (from Yahoo Finance).
            :param old_first(bool): If True, the oldest observation is in the top of the table.
            :param sep(str): string separating the stock's symbol from the rest of the name of the file.
            the name of the file must follow the pattern SYMBOL + sep + REST.csv
        """
        self.files = os.listdir(path)
        self.data = {}
        self.old_first = old_first
        for f in self.files:
            if "csv" in f:
                file_path = os.path.join(path, f)
                df = pd.read_csv(file_path, index_col = "Date")
                df.sort_index(ascending = old_first, inplace = True)
                symbol = f.split(sep)[0]
                self.data[symbol] = df
                #Rename columns
                newnames = {col:f'{col}_{symbol}' for col in df.columns}
                df.rename(columns = newnames, inplace = True)        
                
    def __getitem__(self, key):
        return self.data[key]
    
    def __len__(self):
        return len(self.data)
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()

    
def norm_hlc_by_open(data: StockData, inplace = False) -> StockData:
    """
    TO DO: Make this function more general (normalize_by_col)
    probably I'll need to use regex and filter method of a dataframe
    """
    if not inplace:
        data = deepcopy(data)

    for symbol in data.keys():
        df = data[symbol]
        
        # Target columns
        tar_cols = [f'High_{symbol}', f'Low_{symbol}', f'Close_{symbol}']
        
        # Normalizing column
        norm_col = df.loc[:, f'Open_{symbol}']
        
        for col in tar_cols:
            df.loc[:, col] = df.loc[:, col] / norm_col
    return data
            
def drop_cols(dataframe: pd.DataFrame, keep: str) -> pd.DataFrame:
    """
    keep is a regex string
    r'(^Open|^Close|^Low|^High)': Keeps Open Close Low High columns
    r'(^Open|Close|^Low|^High)': Keeps Open Close Low High and Adj Close
    """
    return dataframe.filter(regex = keep, axis = 1)

def join(data: StockData) -> pd.DataFrame:
    keys = list(data.keys())
    df = data[keys[0]]
    for i in range(1, len(keys)):
        lsuf = keys[i-1]
        rsuf = keys[i]
        df = df.join(data[keys[i]], lsuffix  = f'_{lsuf}', rsuffix = f'_{rsuf}', on  = 'Date', how = 'left')
    return df

def create_batches(data: np.Array, period_len: int) -> Array:
    for 
    
    

path = './data/stocks'
old_first = True
sep = '_'
stocks = StockData(path, old_first, sep)
norm_stocks = norm_hlc_by_open(stocks, inplace = False)
data = join(norm_stocks)
data = drop_cols(data, keep = r'(^High|^Low|^Close)').to_numpy()
data = jnp.array(data)
