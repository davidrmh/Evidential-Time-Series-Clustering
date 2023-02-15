import os
import jax
import pandas as pd
from typing import Union, Tuple, Optional
from jaxtyping import Float, Array


class StockData:
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
        for f in files:
            if "csv" in f:
                file_path = os.path.join(path, f)
                df = pd.read_csv(file_path, index_col = "Date")
                df.sort_index(ascending = old_first, inplace = True)
                symbol = f.split(sep)[0]
                self.data[symbol] = df
        
    def norm_by_open(self):
        
path = './data/stocks'
old_first = True
sep = '_'
data = StockData(path, old_first, sep)
        
                
                
                
            
        
        
        