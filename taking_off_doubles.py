import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc

import torch
import torch.nn as nn

device = 'cpu'

#Historical data ==> data_without_doubles

#Path to historical data
path_data = r"C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\Forecast-air-pollution\data\historical_data.csv"

data_tot = pd.read_csv(path_data, parse_dates=[0])  

# Order dataframe according to the dates
data_ordered = data_tot.sort_values(by=data_tot.columns[0])
data = pd.DataFrame(data_ordered)

data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)

#Il y a certains capteurs qui sont redondants donc on va créer un nouveau fichier 
# avec seulement les données qu'on veut, sans les redondances, 
# et avec les capteurs qui comblent les erreurs

#Capteurs inutiles, cf le doc GNNs
redondants = ['CLDP0019', 'CLDP0030', 'CLDP0097', 'CLDP0154', 'CLDP0296', 'CLDP0016', 'CLDP0264', 'CLDP0545', 'CLDP0546', 'CLDP0549', 'CLDP0031']
extraction = []

#On commence par virer les capteurs qui servent à rien
for col in data.columns : 
    if col not in redondants : 
        extraction.append(col)

data_new = data[extraction]

#Création d'un nouveau fichier
data_new.to_csv(r"C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\Forecast-air-pollution\data\data_without_double.csv"
, index=True)
