import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc

import torch
import torch.nn as nn

device = 'cpu'

#Path to data_clean
path_data = r"C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\Forecast-air-pollution\data\data_clean.csv"


data_tot = pd.read_csv(path_data, parse_dates=[0]) 
df = pd.DataFrame(data_tot)

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

def filling_beginning(df) : 
    df_filled = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for col in df.columns :
        
        #index of the lign
        i = 0
        #list of the values non nan to calculate the mean later
        futur_values = []
        
        while i < len(df)-1 and df.iloc[i][col]==df.iloc[i+1][col]:
            i += 1
        
        #end when the sensor is activated
        print("beginning = ", i)
        
        #We calculate the mean of the following values
        for k in range(i, len(df[col])) : 
            if pd.isna(df.iloc[k][col]) == False :
                futur_values.append(df.iloc[k] [col])
                df_filled.iloc[k][col] = df.iloc[k][col]
        
                
        mean = sum(futur_values)/len(futur_values)

        #We fill the nan with the average of these futur values
        for j in range (i) : 
            df_filled.iloc [j, df.columns.get_loc(col)] = mean               
            
    
    return df_filled

df_filled = filling_beginning(df)


def check_for_nan(df) : 
    return df.isnull().values.any()

has_nan = check_for_nan(df_filled)
print(has_nan)

df_filled.to_csv(r"C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\Forecast-air-pollution\data\data_all_filled.csv")