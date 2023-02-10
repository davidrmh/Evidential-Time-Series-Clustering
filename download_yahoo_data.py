# FOR MY OWN USE
from datetime import datetime
import pandas as pd
import os

symbols = pd.read_csv('./data/sp500_symbols.csv')
#Periods Unix timestamp (seconds since January 1, 1970)
str_start_date = '01-01-2015'
t_start_date = datetime.strptime(str_start_date, '%m-%d-%Y')
str_end_date = '01-31-2023'
t_end_date = datetime.strptime(str_end_date, '%m-%d-%Y')
epoch_date = datetime.strptime('01-01-1970', '%m-%d-%Y')

period1 = int( (t_start_date - epoch_date).total_seconds() )
period2 = int( (t_end_date - epoch_date).total_seconds() )

for i in range(symbols.shape[0]):
    s = symbols.iloc[i][0]
    file_path = f'./data/stocks/{s}_{str_start_date}_{str_end_date}.csv' #Need to improve with a given path
    query = f'https://query1.finance.yahoo.com/v7/finance/download/{s}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true'
    try:
        data = pd.read_csv(query)
        data.to_csv(file_path, index = False)
    except:
        print(f'Error for {s}')
        continue
    
    
