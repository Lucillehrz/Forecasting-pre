import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch.nn import GRUCell
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc


device= 'cpu'

#Downloading data
path_data = r'C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\forecasting\forecasting\data\data_sensors_activated\data_200_all_activated.csv'
path_loca = r'C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\Forecast-air-pollution\data\sensors_info.json'

data_measures = pd.read_csv(path_data, parse_dates=[0]) 
data_loca = pd.read_json(path_loca)

df_data = pd.DataFrame(data_measures) 
df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
df_data.set_index('Timestamp', inplace=True)
data = df_data.values

#Log and normalization
data = np.log(data + 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

df_loca = pd.DataFrame(data_loca)
# Defining the index
df_loca.set_index('SiteCode', inplace=True)

def calcul_distance(sensor_1, sensor_2) : 
    loc_sensor_1 = df_loca.loc[sensor_1, ['Latitude', 'Longitude']].values
    loc_sensor_2 = df_loca.loc[sensor_2, ['Latitude', 'Longitude']].values
    
    loc_sensor_1 = torch.tensor(loc_sensor_1, dtype=torch.float)
    loc_sensor_2 = torch.tensor(loc_sensor_2, dtype=torch.float)
    
    # calculating the euclidian distance between both sensors
    distance = torch.dist(loc_sensor_1, loc_sensor_2, p=2)
    
    return distance.item()  

def normalisation(liste) : 
    min_val = np.min(liste)
    max_val = np.max(liste)
    scaled_liste = (liste - min_val) / (max_val - min_val)
    return scaled_liste

def gaussian_function(liste, sigma=1.0) : 
    liste = np.array(liste)
    gaussian_values = np.exp(-(liste ** 2) / (2 * sigma ** 2))
    return gaussian_values

from scipy.stats import pearsonr

#Pearson correaltion
def calcul_correlation(sensor_1, sensor_2) : 
    serie1 = np.array(df_data[sensor_1].values)
    serie2 = np.array(df_data[sensor_2].values)
    
    corr_coef, _ = pearsonr(serie1, serie2)
    return corr_coef 

def calculate_edges_matrix(measures) : 
    num_nodes = measures.shape[1]
    edge_index = []
    edge_weight = []
    
    for sensor_1 in measures.columns : 
        sensor_1_index = measures.columns.get_loc(sensor_1)
        
        for sensor_2 in measures.columns : 
            sensor_2_index = measures.columns.get_loc(sensor_2)
            
            if sensor_1 != sensor_2 : 
                distance = calcul_distance(sensor_1, sensor_2)
                if distance < 0.1 : 
                    edge_index.append([sensor_1_index, sensor_2_index])
                    edge_weight.append(distance)
                
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # Trensposing to get the shape [2, num_edges]
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            
    return edge_index, edge_weight   

sigma = 1

def calculate_edges_matrix_v2(measures) : 
    num_nodes = measures.shape[1]
    edge_index = []
    edge_weight = []
    distance_tot = []
    indices_tot = []
    
    for sensor_1 in measures.columns : 
        sensor_1_index = measures.columns.get_loc(sensor_1)
        
        for sensor_2 in measures.columns : 
            sensor_2_index = measures.columns.get_loc(sensor_2)
            
            if sensor_1 != sensor_2 : 
                distance = calcul_distance(sensor_1, sensor_2)
                if distance < 0.05 : 
                    edge_index.append([sensor_1_index, sensor_2_index])
                    edge_weight.append(distance)
    print(type(edge_weight))            
    edge_weight = gaussian_function(edge_weight, sigma)          
                
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            
    return edge_index, edge_weight

def calculate_edges_matrix_v3(measures, alpha=0.5, beta=0.5) : 
    num_nodes = measures.shape[1]
    edge_index = []
    edge_weight = []
    distance_tot = []
    indices_tot = []
    
    for sensor_1 in measures.columns : 
        sensor_1_index = measures.columns.get_loc(sensor_1)
        
        for sensor_2 in measures.columns : 
            sensor_2_index = measures.columns.get_loc(sensor_2)
            
            if sensor_1 != sensor_2 : 
                distance = calcul_distance(sensor_1, sensor_2)
                correlation = calcul_correlation(sensor_1, sensor_2)
                if distance != 0 : 
                    coef = alpha*(1/distance) + beta*correlation 
                    if coef > 0.5 : 
                        edge_index.append([sensor_1_index, sensor_2_index])
                        edge_weight.append(coef)
    print(type(edge_weight))            
    #edge_weight = gaussian_function(edge_weight, sigma)    
    edge_weight = normalisation(edge_weight)      
                
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  #
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            
    return edge_index, edge_weight

edge_index, edge_weight = calculate_edges_matrix_v3(df_data)
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)

num_nodes = df_data.shape[1]
num_time_steps = 1  #sequence lentgh
num_features_per_node = 1 
batch_size = 32
hidden_dim = 64
learning_rate = 0.0001
num_epochs = 4
n = 5

training = True

def create_sequences(data, seq_length, n):
    xs = []
    ys = []
    for i in range(len(data) - seq_length -n+1):
        x = data[i:i+seq_length].transpose(1,0)
        y = data[i+seq_length : i+seq_length+n]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(data, num_time_steps, n)
#X is of shape (num_sample, num_time_steps, num_nodes)
#y is of shape (num_samples, num_nodes

#Splitting training and testing data
split_index = int(len(X)*0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(X_train.shape)


graph_list_train = []

for i in range(X_train.shape[0]) : 
    x = X_train[i]
    y = y_train[i]
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
    graph_list_train.append(data)

train_loader = DataLoader(graph_list_train, batch_size, shuffle=True)

graph_list_test = []

for i in range(X_test.shape[0]) : 
    x = X_test[i]
    y = y_test[i]
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
    graph_list_test.append(data)

test_loader = DataLoader(graph_list_test, batch_size, shuffle=False)



#shaoe of the input X : [num_sample, num_nodes, num_time_steps]
class GraphRNN(nn.Module):
    def __init__(self, num_features_per_node, num_time_steps, hidden_dim,prediction_steps=n, num_layers=2):
        super(GraphRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_features_per_node = num_features_per_node
        self.num_time_steps = num_time_steps
        self.predictions_step = prediction_steps
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features_per_node * num_time_steps, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        
        #GRUCell input has a shape (num_nodes * hidden_dim)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, prediction_steps)

    def forward(self, x, edge_index, edge_weight, h):

        num_samples = x.size()[0]
        
        #Graph convolution
        x = x.reshape(num_samples * num_nodes, num_time_steps)
  
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
        
        x = x.reshape(num_samples * num_nodes, -1)
        
       #Updating hidden state with gru
        h = h.reshape(num_samples * num_nodes, self.hidden_dim)
        h = self.gru(x, h)
        
        #Final prediction
        out = self.fc(h)
        out = out.reshape(num_samples, num_nodes, self.predictions_step)
        h = h.reshape(num_samples, num_nodes, self.hidden_dim)
        
        return out, h
    

model = GraphRNN(num_features_per_node, num_time_steps, hidden_dim, n)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

#num_sample_tot = X.shape[0]
X = torch.tensor(X, dtype=torch.float32)
h = torch.zeros((X.shape[0], num_nodes, hidden_dim)).to(device)

print(X.shape, edge_index.shape, edge_weight.shape, h.shape)
output, h = model(X, edge_index, edge_weight, h)

def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    #for batch in train_loader :
    for batch_index, batch in enumerate(train_loader):
        x = batch.x.reshape(-1, num_nodes, num_time_steps)
        h = torch.zeros(x.size(0), num_nodes, hidden_dim).to(device)
        output, h = model(x, batch.edge_index, batch.edge_weight, h)

        loss = loss_function(output, batch.y.reshape(output.shape))
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
            
    print()
    

def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch in test_loader:
        with torch.no_grad():
            x = batch.x.reshape(-1, num_nodes, num_time_steps)
            h = torch.zeros(x.size(0), num_nodes, hidden_dim).to(device)
            #h = torch.zeros(batch.x.size(0), num_nodes, hidden_dim).to(device)
            output, h = model(x, edge_index, edge_weight, h)
            output = output.squeeze(2)
            loss = loss_function(output, batch.y.reshape(output.shape))
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)
    
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    
    print()
    
if training : 
    for epoch in range(num_epochs) : 
        train_one_epoch()
        validate_one_epoch()
        
with torch.no_grad():
    h = torch.zeros(X_train.size(0), num_nodes, hidden_dim).to(device)
    training_predictions, h = model(X_train.to(device), edge_index, edge_weight, h)
    
    h = torch.zeros(X_test.size(0), num_nodes, hidden_dim).to(device)
    testing_predictions, h = model(X_test.to(device), edge_index, edge_weight, h)

name_saving = 't+n.pth'
FILE = f'C:/Users/Lucille/OneDrive/Documents/Forecasting air pollution/Forecast-air-pollution/model_avec_graphes/models/{name_saving}'
torch.save(model.state_dict(), FILE)

def MAPE(y_true, y_pred):
    
    if len(y_true) != len(y_pred):
        raise ValueError("Les séries doivent avoir la même longueur.")

    #Avoiding dividing by 0
    mask = y_true != 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    if isinstance(y_true_filtered, torch.Tensor):
        y_true_filtered = y_true_filtered.cpu().numpy()
    if isinstance(y_pred_filtered, torch.Tensor):
        y_pred_filtered = y_pred_filtered.cpu().numpy()
    
    mape = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered) * 100
    mape = np.mean(mape) 
    return mape

def predictions_at_m(m) : 
    training_predictions_at_m = training_predictions[:,:,m-1]
    training_predictions_at_m = training_predictions_at_m.reshape(-1,num_nodes)
    
    testing_predictions_at_m = testing_predictions[:,:,m-1]
    testing_predictions_at_m = testing_predictions_at_m.reshape(-1,num_nodes)
    
    #denormaization
    training_predictions_at_m = scaler.inverse_transform(training_predictions_at_m)
    new_y_train_m = scaler.inverse_transform(y_train[:,m-1,:])

    testing_predictions_at_m = scaler.inverse_transform(testing_predictions_at_m)
    new_y_test_m = scaler.inverse_transform(y_test[:,m-1,:])
    
    training_predictions_at_m = np.exp(training_predictions_at_m) - 1
    new_y_train_m = np.exp(new_y_train_m) - 1

    testing_predictions_at_m = np.exp(testing_predictions_at_m) - 1
    new_y_test_m = np.exp(new_y_test_m) - 1

    #Plot of  a random slot on a random sensor
    i=20
    plt.plot(new_y_train_m[4100:4300,i], label='Actual value')
    plt.plot(training_predictions_at_m[4100+m:4300+m,i], label='Predicted value', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('Air pollution')
    plt.legend()
    plt.show()
    
    plt.plot(new_y_test_m[2100:2300,i], label='Actual value')
    plt.plot(testing_predictions_at_m[2100+m:2300+m,i], label='Predicted value', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('Air pollution')
    plt.legend()
    plt.show()
    
    mape_train = MAPE(new_y_train_m[:-m], training_predictions_at_m[m:])
    print("mape tot training = ", mape_train)

    mape_test = MAPE(new_y_test_m[:-m], testing_predictions_at_m[m:])
    print("mape tot testing = ", mape_test)
    
    
predictions_at_m(2)
if n>=5 : 
    predictions_at_m(5)
    
if n>=10 : 
    predictions_at_m(10)

    
    