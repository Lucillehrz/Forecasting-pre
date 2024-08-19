import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc

import torch
import torch.nn as nn

device = 'cpu'

nb_sensors = 100

path = r"C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\Forecast-air-pollution\data\data_clean.csv"

data_tot = pd.read_csv(path, parse_dates=[0])  

# Order dataframe according to the dates
data_ordered = data_tot.sort_values(by=data_tot.columns[0])
df = pd.DataFrame(data_ordered)

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

#What is the size of a column? 
length = len(df['CLDP0029'])
print('One column has a length of', length, 'which means from the', df.index[0], 'to the', df.index[length-1])

#Let's adapt the number of sensors so that we have at least more than one year of data for training


#Finding the timestamp from which all sensors are activated
def finding_begining(df, nb_sensors) : 
    beg = 0
    for col in df.columns[:nb_sensors + 1] : 
        #if col not in ['CLDP0046', 'CLDP0101']:
            i = 0
            while i < len(df) and df.iloc[i][col]==df.iloc[i+1][col]:
                i += 1
            
            if i > beg : 
                beg = i 
    
    print(f"The first {nb_sensors} are activated from", beg)
    print("Which corresponds to the date", df.index[beg])
    return beg
        
beginning  = finding_begining(df, nb_sensors)       

#on vire les lignes en-dessous
data_extracted = dc(df.iloc[beginning:, :nb_sensors+1])

size_data = length - beginning 
print('We would have', size_data, 'pieces of data')

def check_for_nan(df) : 
    return df.isnull().values.any()

has_nan = check_for_nan(data_extracted)
print(has_nan)

data_extracted.to_csv(f'C:/Users/Lucille/OneDrive/Documents/Forecasting air pollution/Forecast-air-pollution/data/data_sensors_activated/data_{nb_sensors}_all_activated.csv', index=True)