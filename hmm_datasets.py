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

def create_batches(data: Array, period_len: int, old_first: bool = True) -> Array:
    """
    data is an array with shape (total_obs, emission_dim).
    when old_first is True the oldest observation is the first observation in the data.
    The result is an array with shape (batches, period_len, emission_dim).
    
    """
    
    total_steps, obs_dim = data.shape
    
    # Bottom up approach (oldest observation is in the top of the table)
    if old_first:
        
        # First batch is obtained manually
        batch_count = 1
        start_idx = total_steps - batch_count * period_len
        end_idx = total_steps - (batch_count - 1) * period_len
        
        batches = data[start_idx:end_idx, :]
        batches = batches.reshape((1, *batches.shape))
        
        batch_count = batch_count + 1
        start_idx = total_steps - batch_count * period_len
        end_idx = total_steps - (batch_count - 1) * period_len
        
        while start_idx >= 0:
            # Get the new batch
            tmp = data[start_idx:end_idx, :]
            
            # Reshape
            tmp = tmp.reshape((1, *tmp.shape))
            
            # Concatenate with previous batches (axis = 0)
            batches = jnp.concatenate((batches, tmp), axis = 0)
            
            # Update indexes
            batch_count = batch_count + 1
            start_idx = total_steps - batch_count * period_len
            end_idx = total_steps - (batch_count - 1) * period_len
    
        return batches
    
    # Top to bottom approach (newest observation is in the top of the table)
    
    # First batch is obtained manually
    batch_count = 1
    start_idx = (batch_count - 1) * period_len
    end_idx = batch_count * period_len
        
    batches = data[start_idx:end_idx, :]
    batches = batches.reshape((1, *batches.shape))
        
    batch_count = batch_count + 1
    start_idx = (batch_count - 1) * period_len
    end_idx = batch_count * period_len
        
    while end_idx < total_steps:
        # Get the new batch
        tmp = data[start_idx:end_idx, :]
            
        # Reshape
        tmp = tmp.reshape((1, *tmp.shape))
            
        # Concatenate with previous batches (axis = 0)
        batches = jnp.concatenate((batches, tmp), axis = 0)
            
        # Update indexes
        batch_count = batch_count + 1
        start_idx = (batch_count - 1) * period_len
        end_idx = batch_count * period_len
        
    return batches 

def batches_norm_hlc_by_open(stocks: StockData, period_len: int = 15):
    
    # Normalize data by diving over open price
    norm_stocks = norm_hlc_by_open(stocks, inplace = False)
    
    # Join tables
    data = join(norm_stocks)
    
    # Drop unnecesary columns
    data = drop_cols(data, keep = r'(^High|^Low|^Close)').to_numpy()
    data = jnp.array(data)
    
    # Get batches
    batches = create_batches(data, period_len, stocks.old_first)
    
    return batches
    


path = './data/stocks'
old_first = True
sep = '_'
stocks = StockData(path, old_first, sep)
period_len = 15
batches_hlc = batches_norm_hlc_by_open(stocks, period_len)

