import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc

import torch
import torch.nn as nn

device = 'cpu'

path = r"C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\Forecast-air-pollution\data\data_all_filed.csv"

data_tot = pd.read_csv(path, parse_dates=[0])  

# Order dataframe according to the dates
data_ordered = data_tot.sort_values(by=data_tot.columns[0])
df = pd.DataFrame(data_ordered)

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

nb_sensors = 100

list_sensors = []

for i in range(nb_sensors) : 
    list_sensors.append(df.columns[i])

data_extracted = df[list_sensors]

data_extracted.to_csv(f'C:/Users/Lucille/OneDrive/Documents/Forecasting air pollution/Forecast-air-pollution/data/data_since_always/data_{nb_sensors}.csv', index=True)