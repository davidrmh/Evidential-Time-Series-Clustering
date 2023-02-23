import os
import jax.numpy as jnp
import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional
from jaxtyping import Float, Array
from copy import deepcopy
import pickle
from datetime import datetime

class StockData(dict):
    def __init__(self,
                 path: str,
                 start: str = '2015-01-01',
                 end: str = '2023-01-31',
                 old_first: bool = True, 
                 sep: str = "_"):
        """
        Args:
            :param path(str): path of the directory containing the csv file with stock prices (from Yahoo Finance).
            :param old_first(bool): If True, the oldest observation is in the top of the table.
            :param start(str): String in format YYYY-MM-DD. Start date.
            :param end(str): String in format YYYY-MM-DD. End date.
            :param sep(str): string separating the stock's symbol from the rest of the name of the file.
            the name of the file must follow the pattern SYMBOL + sep + REST.csv
        """
        self.files = os.listdir(path)
        self.data = {}
        self.old_first = old_first
        for f in self.files:
            if "csv" in f:
                # Read CSV
                file_path = os.path.join(path, f)
                df = pd.read_csv(file_path, index_col = "Date")
                
                # Sort date according to old_first
                df.sort_index(ascending = old_first, inplace = True)
                
                # Filter data in date ranges
                df = df.loc[start:end]
                
                # Rename columns
                symbol = f.split(sep)[0]
                newnames = {col:f'{col}_{symbol}' for col in df.columns}
                df.rename(columns = newnames, inplace = True)
                self.data[symbol] = df
                

    def get_log_returns(self):
        """
        Obtain the log returns for each stock series.
        Adj Close prices are used

        Returns
        -------
        Dictionary with pandas series containing the returns.

        """
        dict_ret = {}
        for s in self.keys():    
            # For simplicity sorts the data from oldest to newest
            d = self[s]
            d = d.sort_index(ascending = True)
            nobs = d.shape[0]
            dates = d.index.values
            
            # Use Adj Close prices
            d = d.filter(regex = r"^Adj", axis = 1).to_numpy()
            ret = np.log(d[1:nobs]) - np.log(d[0:nobs - 1])
            ret = pd.DataFrame(ret.ravel(), index = dates[1:nobs])
            ret = ret.rename_axis(index = "Date")
            ret = ret.rename(columns = {0: f'Adj Close_{s}'})
            dict_ret[s] = ret
        return dict_ret
                
    def __getitem__(self, key):
        return self.data[key]
    
    def __len__(self):
        return len(self.data)
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()

def join(data) -> pd.DataFrame:
    """
    Join data in a DataFrame.

    Parameters
    ----------
    data : Dictionary like object. It must implement key and
    __getitem__ methods.

    Returns
    -------
    df : pandas DataFrame
        Columns from each Dataframe joined side by side.

    """
    keys = list(data.keys())
    df = data[keys[0]]
    for i in range(1, len(keys)):
        lsuf = keys[i-1]
        rsuf = keys[i]
        df = df.join(data[keys[i]],
                     lsuffix  = f'_{lsuf}',
                     rsuffix = f'_{rsuf}',
                     on  = 'Date',
                     how = 'left')
    return df

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
    
def make_folds(data: Array, e: int = 0):
    """
    Create folds for leave-one batch out cross-validation

    Parameters
    ----------
    data : Array
        Array with shape (batches, len_period, emi_dim).
    e : int, optional
        Length of embargo period. This applies to
        axis 1 of array. Indicates the number of periods
        to ignore from the left and from the right. The default is 0.

    Raises
    ------
    Exception
        If e >= len_period - 1 all the data is discarded. 

    Returns
    -------
    list with tuples. The first entry of each tuple is a training data set
    the second entry is a test data set.

    """
    len_period = data.shape[1]
    nbatch = data.shape[0]
    idx = list(range(0, e)) + list(range(len_period - 1, len_period - 1 - e, -1))
    idx = jnp.array(idx)
    idx = jnp.unique(idx)
    if len(idx) >= len_period:
        raise Exception(f"Embargo parameter e = {e} is too large. There's no data left")
    
    folds = []
    for i in range(1, nbatch + 1):
        train = jnp.concatenate([data[0:nbatch - i], data[nbatch -i + 1:]])
        test = data[nbatch -i]
        # Embargo some data
        if e > 0:
            test = jnp.delete(test, idx, axis = 0)
        folds.append( (train,  test) )
    return folds

def save_checkpoint(obj, dirpath = "./experiments", fname="experiment"):
    cwd = os.getcwd()
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    fname = os.path.join(dirpath, fname) + '.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)
        print('\n Object saved \n')

def load_checkpoint(filepath):
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj
       