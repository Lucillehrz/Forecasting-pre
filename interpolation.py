import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc

import torch
import torch.nn as nn

device = 'cpu'
path_data = r"C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\Forecast-air-pollution\data\data_without_double.csv"


data_tot = pd.read_csv(path_data, parse_dates=[0]) 
df = pd.DataFrame(data_tot)

def filling_linear(df) : 
    df_filled = dc(df.interpolate(method='linear', limit_direction='both'))
    return df_filled

df_filled = filling_linear(df)

df_filled['Timestamp'] = pd.to_datetime(df_filled['Timestamp'])
df_filled.set_index('Timestamp', inplace=True)

df_filled.to_csv(r"C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\Forecast-air-pollution\data\data_clean.csv")