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
import os

from torch.utils.tensorboard import SummaryWriter
import torchvision
import datetime

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(f"Using device: {device}")


#Downloading data
#path_data ='/scratch_dgxl/lhuriez/forecasting/data/data_sensors_activated/data_400_all_activated.csv'
#path_loca ='/scratch_dgxl/lhuriez/forecasting/data/sensors_info.json'
path_data =r'C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\forecasting\forecasting\data\data_sensors_activated\data_200_all_activated.csv'
path_loca =r'C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\forecasting\forecasting\data\sensors_info.json'
    

print("data :", path_data)

#output_dir = f'/scratch_dgxl/lhuriez/forecasting/Results/graph/GRUCell_GCN' 
#output_dir = r'C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\forecasting\forecasting\Results\graph\GRUCell_GCN'
output_dir = r'C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\forecasting\forecasting\Results\graph\GRU_t+n'

data_measures = pd.read_csv(path_data, parse_dates=[0]) 
data_loca = pd.read_json(path_loca)

#Operation done only with 300 sensors, maybe there are similar later with the same problem

df_data = pd.DataFrame(data_measures) 

# Identify columns with constant values
constant_columns = [col for col in df_data.columns if df_data[col].nunique() == 1]
# Drop these columns
df_data = df_data.drop(columns=constant_columns)
print("Columns with constant values removed:", constant_columns)


df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
df_data.set_index('Timestamp', inplace=True)

data = df_data.values

num_nodes = df_data.shape[1]
num_time_steps = 1  
num_features_per_node = 1 
num_layers_conv = 3
batch_size = 64
hidden_dim = 64
learning_rate = 0.0001
num_epochs = 80
seuil = 0.5
alpha = 0.5
beta = 0.5

training = True 
saving = True

print('sequence_length =', num_time_steps)
print('hidden_size =', hidden_dim)
print('num_layers_conv =', num_layers_conv)
print('learning_rate =', learning_rate)
print('num_epochs =', num_epochs)
print('batch_size =', batch_size)
print('num_nodes =', num_nodes)
print("sole = ", seuil)
print("alpha, beta =", alpha, beta)

#Logarithm and normalization
data = np.log(data + 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
df_loca = pd.DataFrame(data_loca)

# Defining the SiteCode as the index
df_loca.set_index('SiteCode', inplace=True)

def cal_distance(sensor_1, sensor_2) : 
    loc_sensor_1 = df_loca.loc[sensor_1, ['Latitude', 'Longitude']].values
    loc_sensor_2 = df_loca.loc[sensor_2, ['Latitude', 'Longitude']].values
    
    # Converting to tensors
    loc_sensor_1 = torch.tensor(loc_sensor_1, dtype=torch.float)
    loc_sensor_2 = torch.tensor(loc_sensor_2, dtype=torch.float)
    
    #Calculating the euclidian distance between two sensors
    distance = torch.dist(loc_sensor_1, loc_sensor_2, p=2)
    
    return distance.item() 

def normalization(liste) : 
    min_val = np.min(liste)
    max_val = np.max(liste)
    scaled_liste = (liste - min_val) / (max_val - min_val)
    return scaled_liste

def gaussian_function(liste, sigma=1.0) : 
    liste = np.array(liste)
    gaussian_values = np.exp(-(liste ** 2) / (2 * sigma ** 2))
    return gaussian_values

from scipy.stats import pearsonr

#Pearson correlation
def cal_correlation(sensor_1, sensor_2) : 
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
                distance = cal_distance(sensor_1, sensor_2)
                if distance < 0.1 : 
                    edge_index.append([sensor_1_index, sensor_2_index])
                    edge_weight.append(distance)
                
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # Transposing to get the shape [2, num_edges]
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            
    return edge_index, edge_weight   

sigma = 1

def calculate_edges_matrix_v2(measures) : 
    num_nodes = measures.shape[1]
    edge_index = []
    edge_weight = []
    distance_tot = []
    
    for sensor_1 in measures.columns : 
        sensor_1_index = measures.columns.get_loc(sensor_1)
        
        for sensor_2 in measures.columns : 
            sensor_2_index = measures.columns.get_loc(sensor_2)
            
            if sensor_1 != sensor_2 : 
                distance = cal_distance(sensor_1, sensor_2)
                if distance < 0.05 : 
                    edge_index.append([sensor_1_index, sensor_2_index])
                    edge_weight.append(distance)
           
    edge_weight = gaussian_function(edge_weight, sigma)          
                
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # Transposing to get the shape [2, num_edges]
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            
    return edge_index, edge_weight

def calculate_edges_matrix_v3(measures, alpha=alpha, beta=beta) : 
    num_nodes = measures.shape[1]
    edge_index = []
    edge_weight = []
    distance_tot = []
    
    for sensor_1 in measures.columns : 
        sensor_1_index = measures.columns.get_loc(sensor_1)
        
        for sensor_2 in measures.columns : 
            sensor_2_index = measures.columns.get_loc(sensor_2)
            
            if sensor_1 != sensor_2 : 
                distance = cal_distance(sensor_1, sensor_2)
                correlation = cal_correlation(sensor_1, sensor_2)
                if distance != 0 : 
                    coef = alpha*(1/distance) + beta*correlation 
                    if coef > seuil : 
                        edge_index.append([sensor_1_index, sensor_2_index])
                        edge_weight.append(coef)
   
    edge_weight = normalization(edge_weight)      
                
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # # Transposing to get the shape [2, num_edges]
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            
    return edge_index, edge_weight

edge_index, edge_weight = calculate_edges_matrix_v3(df_data)
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)



def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length].transpose(1,0)
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(data, num_time_steps)
#X is of shape (num_sample, num_time_steps, num_nodes)
#y is of shape (num_samples, num_nodes

#Splitting training data from testing data
split_index = int(len(X)*0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# Converting data into torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


#Creating the dataloaders for the batches
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


#Shape of X in the input : [num_sample, num_nodes, num_time_steps]
class GraphRNN(nn.Module):
    def __init__(self, num_features_per_node=num_features_per_node, num_time_steps=num_time_steps, hidden_dim=hidden_dim, num_layers=num_layers_conv):
        super(GraphRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_features_per_node = num_features_per_node
        self.num_time_steps = num_time_steps
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features_per_node * num_time_steps, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        
        #GRUCell takes an input with the shape (num_nodes * hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight, h):

        num_samples = x.size()[0]
        x = x.reshape(num_samples * num_nodes, num_time_steps)
        
        #Convolution
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))

        x = x.reshape(num_samples * num_nodes, -1)
        
       #Updating hidden states with GRU
        h = h.reshape(num_samples * num_nodes, self.hidden_dim)
        h = self.gru(x, h)
        
        #Final prediction
        out = self.fc(h)
        out = out.reshape(num_samples, num_nodes, -1).to(device)
        h = h.reshape(num_samples, num_nodes, -1).to(device)
        
        return out, h
    
    
model = GraphRNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

training_loss = []
training_accuracy = []

validation_loss = []
validation_accuracy = []

#For tensorboard
#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#writer = SummaryWriter(log_dir=log_dir)

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


def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    
    train_loss = 0.0
    train_mape = 0.0

    #for batch in train_loader :
    for batch_index, batch in enumerate(train_loader):
        x = batch.x.reshape(-1, num_nodes, num_time_steps)
        h = torch.zeros(x.size(0), num_nodes, hidden_dim).to(device)
        output, h = model(x, batch.edge_index, batch.edge_weight, h)

        loss = loss_function(output, batch.y.reshape(output.shape))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        targets = batch.y.reshape(output.shape).cpu().numpy()
        outputs = output.detach().cpu().numpy()
        train_mape += MAPE(targets, outputs)
    
    #Version where we add value in the graph at each epoch
    train_mape = train_mape/len(train_loader)
    train_loss = train_loss/len(train_loader)
    
    
    training_loss.append(train_loss)
    training_accuracy.append(train_mape)
    
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_mape:.2f}%')

    
    
def validate_one_epoch():
    model.train(False)
    running_loss = 0.0
    running_mape = 0.0

    for batch in test_loader:
        with torch.no_grad():
            x = batch.x.reshape(-1, num_nodes, num_time_steps)
            h = torch.zeros(x.size(0), num_nodes, hidden_dim).to(device)
            #h = torch.zeros(batch.x.size(0), num_nodes, hidden_dim).to(device)
            output, h = model(x, edge_index, edge_weight, h)
            output = output.squeeze(2)
            loss = loss_function(output, batch.y.reshape(output.shape))
            targets = batch.y.reshape(output.shape)
            running_loss += loss.item()
            running_mape += MAPE(targets, output)

    avg_loss_across_batches = running_loss / len(test_loader)
    avg_mape_across_batches = running_mape /len(test_loader)
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches), 'Avg mape: ', avg_mape_across_batches)
    print('***************************************************')
    
    validation_loss.append(avg_loss_across_batches)
    validation_accuracy.append(avg_mape_across_batches)
    #writer.add_scalar('Validation Loss', avg_loss_across_batches, epoch)
    #writer.add_scalar('Validation accuracy', avg_mape_across_batches , epoch)


if training : 
    for epoch in range(num_epochs) : 
        train_one_epoch()
        validate_one_epoch()

#For tensorboard
#writer.close()




#Predicting the training set
with torch.no_grad():
    h = torch.zeros(X_train.size(0), num_nodes, hidden_dim).to(device)
    #h = torch.zeros(X_train.size(0), num_nodes, hidden_dim).to(device)
    training_predictions, h = model(X_train.to(device), edge_index, edge_weight, h)
    training_predictions = training_predictions.squeeze(2).to('cpu').numpy()
    
#De-normalization
training_predictions = scaler.inverse_transform(training_predictions)
new_y_train = scaler.inverse_transform(y_train.to('cpu'))

#De-logarithm (exp)  
training_predictions = np.exp(training_predictions) - 1
new_y_train = np.exp(new_y_train) - 1


#Predicting the testing set
with torch.no_grad():
    h = torch.zeros(X_test.size(0), num_nodes, hidden_dim).to(device)
    testing_predictions, h = model(X_test.to(device), edge_index, edge_weight, h)
    testing_predictions = testing_predictions.squeeze(2).to('cpu').numpy()


testing_predictions = scaler.inverse_transform(testing_predictions)
new_y_test = scaler.inverse_transform(y_test.to('cpu')) 
testing_predictions = np.exp(testing_predictions) - 1
new_y_test = np.exp(new_y_test) - 1



#Average of MAPE on all sensors for the training and the testing set
mape_train = MAPE(new_y_train[:-1,:], training_predictions[1:,:])
print("mape tot training = ", mape_train)

mape_test = MAPE(new_y_test[:-1], testing_predictions[1:])
print("mape tot testing = ", mape_test)


#Ploting MAPE for each sensor at training and testing
MAPE_list_train = []
MAPE_list_test = []
Number = []

for i in range(num_nodes):
    mape_train = MAPE(new_y_train[:-1,i], training_predictions[1:,i])
    mape_test = MAPE(new_y_test[:-1,i], testing_predictions[1:,i])
    if mape_train<20 and mape_test<20 : 
        MAPE_list_train.append(mape_train)
        MAPE_list_test.append(mape_test)
        Number.append(i)
    
plt.bar(Number, MAPE_list_train, label='Training')
plt.bar(Number, MAPE_list_test, label='Testing', alpha=0.5)
plt.xlabel('Sensors')
plt.ylabel('MAPE') 
plt.legend()
plt.savefig(os.path.join(output_dir, 'MAPE.png'))
plt.close()


# Path to the directory where saving images
name_directory = 'GRUCell_GCN/v1'
output_dir = f'/scratch_dgxl/lhuriez/forecasting/Results/graph/GRUCell_GCN' 


#Creation and saving figure
def save_image(fig, filename):
    file_path = os.path.join(output_dir, filename)
    fig.savefig(file_path)
    plt.close(fig)
    print(f'Image saved at {file_path}')


#Loss plot
fig_loss, ax_loss = plt.subplots()

ax_loss.plot(training_loss, label='Training Loss')
ax_loss.plot(validation_loss, label='Validation Loss')

ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.legend()

filename_loss = 'loss_plot.png'
save_image(fig_loss, filename_loss)


#MAPE plot
fig_mape, ax_mape = plt.subplots()

ax_mape.plot(training_accuracy, label='Training MAPE')
ax_mape.plot(validation_accuracy, label='Validation MAPE')

ax_mape.set_xlabel('Epoch')
ax_mape.set_ylabel('MAPE')
ax_mape.legend()

filename_mape = 'mape_plot_evolution.png'

save_image(fig_mape, filename_mape)

#MAPE by sensor
fig_mape_bar, ax_mape_bar = plt.subplots()

ax_mape_bar.bar(Number, MAPE_list_test, label='Testing MAPE')
ax_mape_bar.bar(Number, MAPE_list_train, label='Training MAPE', alpha=0.5)

ax_mape_bar.set_xlabel('Sensors')
ax_mape_bar.set_ylabel('MAPE')
ax_mape_bar.legend()

filename_mape = 'mape_by_sensor.png'

save_image(fig_mape, filename_mape)

def results(i) : 
    print("sensor :", df_data.columns[i])
    
    fig_i, ax_i = plt.subplots()

    ax_i.plot(new_y_test[len(new_y_test)-200:-1,i], label='real values')
    ax_i.plot(testing_predictions[len(new_y_test)-200+1:,i], label='predicted values')

    ax_i.set_xlabel('Time')
    ax_i.set_ylabel('Air pollution')
    ax_i.legend()

    filename_i = f'plot_testing_{i}.png'

    save_image(fig_i, filename_i)
    
    mape_tr = MAPE(new_y_train[:-1,i], training_predictions[1:,i])
    mape_te = MAPE(new_y_test[:-1,i], testing_predictions[1:,i])
    
    print(f"MAPE of the sensor {i} on the training set :", mape_tr)
    print(f"MAPE of the sensor {i} on the testing set :", mape_te)

results(2)
results(99)
results(200)




    

