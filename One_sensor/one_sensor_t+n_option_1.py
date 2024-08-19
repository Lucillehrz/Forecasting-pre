import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(f"Using device: {device}")

#Parameters
sequence_length = 1
input_size = 1
output_size = 1
hidden_size = 10
num_layers = 3
learning_rate = 0.0001
num_epochs = 50

n = 24

batch_size = 32

training = True
saving = False

print('sequence_length =', sequence_length)
print('input_size =', input_size)
print('output_size =', output_size)
print('hidden_size =', hidden_size)
print('num_layers =', num_layers)
print('learning_rate =', learning_rate)
print('num_epochs =', num_epochs)
print('batch_size =', batch_size)

#path = "/scratch_dgxl/lhuriez/Forecast-air-pollution/data/data_v5.csv"
path = r"C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\forecasting\forecasting\data\data_all_filed.csv"
sensor = 'CLDP0029'


data_tot = pd.read_csv(path, parse_dates=[0])  

# Extracting the data of one sensor 
extracted_data = ['Timestamp', sensor]  # Remplacez par les noms réels des colonnes

data_ordered = data_tot.sort_values(by=data_tot.columns[0])
data = pd.DataFrame(data_ordered[extracted_data])
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)
print(data.head())
df_array = data.values

#We only keep data from the moment the sensor is activated
i = 1
while i < len(data) - 1 and data.iloc[i].item() == data.iloc[i + 1].item():
    i += 1
    
df = data.iloc[i:]
df_array = df.values 
print("i = ",i)


#Saving images
# Path to the directory where results will be saved
#output_dir = f'/scratch_dgxl/lhuriez/forecasting/Results/one_sensor_t+n_v2/{sensor}'
output_dir = r'C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\forecasting\forecasting\Results\one_sensor_t+n_v2\CLDP0029'

#Creation and saving figure
def save_image(fig, filename):
    file_path = os.path.join(output_dir, filename)
    fig.savefig(file_path)
    plt.close(fig)
    print(f'Image saved at {file_path}')

#Plot of the evolution of the air pollution at this sensor
fig_sensor_tot, ax_sensor_tot = plt.subplots() 
ax_sensor_tot.plot(df_array, label=f'{sensor}')
ax_sensor_tot.set_xlabel('Time')
ax_sensor_tot.set_ylabel('Air pollution')
#Saving it
save_image(fig_sensor_tot, f'total_{sensor}.png')

#Plot of air pollution at the sensor only from b1 to b2
b1 = 20000
b2 = 20300
fig_sensor_part, ax_sensor_part = plt.subplots() 
ax_sensor_part.plot(df_array[b1:b2], label=f'{sensor}')
ax_sensor_part.set_xlabel('Time')
ax_sensor_part.set_ylabel('Air pollution')
#Saving it
save_image(fig_sensor_part, f'partial_{sensor}.png')

#Log and normalization
df_array = np.log(df_array + 1)
scaler = MinMaxScaler(feature_range=(0, 1))
df_array = scaler.fit_transform(df_array)


#Functions definition
#Creating the sequences of data for each sample
def create_sequences(data, seq_length, n_steps):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - n_steps + 1):
        x = data[i:i+seq_length]
        y = data[i+seq_length:i+seq_length+n_steps]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    
#Definition of the model    
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_steps):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        self.n_steps = n_steps
        
        self.output_size = n_steps
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=self.bidirectional)
        
        self.fc = nn.Linear(hidden_size * 2, self.output_size) if self.bidirectional else nn.Linear(hidden_size, output_size)
        

    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)

        out, (hn, cn) = self.lstm(x)
        #print("taille our avant fc", out.shape)
        out = self.fc(out[:, -1, :])
    
        return out
        

#Calculation of MAPE, which is our error measure    
def MAPE(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Les séries doivent avoir la même longueur.")

    mask = y_true != 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    # Assurez-vous que y_true_filtered et y_pred_filtered sont des tenseurs
    if isinstance(y_true_filtered, torch.Tensor):
        mape = torch.mean(torch.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    else:
        mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100

    return mape.item() if isinstance(mape, torch.Tensor) else mape


# Creating data sequences
X, y = create_sequences(df_array, sequence_length, n)

X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

split_index = int(len(X)*0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#Initialization of the model
model = LSTMPredictor(input_size, hidden_size, num_layers, n).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#List in whihc we will store accuracy and loss of the training and validation
training_loss = []
training_accuracy = []

validation_loss = []
validation_accuracy = []

#Training one epoch function
def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    train_loss = 0.0
    train_mape = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        output = model(x_batch)
        loss = loss_function(output, y_batch.reshape(output.shape))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        targets = y_batch.reshape(output.shape).cpu().numpy()
        outputs = output.detach().cpu().numpy()
        train_mape += MAPE(targets, outputs)
    
    #Version where we add value in the graph at each epoch
    train_mape = train_mape/len(train_loader)
    train_loss = train_loss/len(train_loader)
    
    #Here, it is the average from t+1 to t+n
    training_loss.append(train_loss)
    training_accuracy.append(train_mape)
    
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_mape:.2f}%')
   
#Validation function for each epoch    
def validate_one_epoch():
    model.train(False)
    running_loss = 0.0
    running_mape = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch.reshape(output.shape))
            targets = y_batch.reshape(output.shape).cpu().numpy()
            outputs = output.detach().cpu().numpy()
            running_loss += loss.item()
            running_mape += MAPE(targets, outputs)

    avg_loss_across_batches = running_loss / len(test_loader)
    avg_mape_across_batches = running_mape /len(test_loader)
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches), 'Avg mape: ', avg_mape_across_batches)
    print('***************************************************')
    
    validation_loss.append(avg_loss_across_batches)
    validation_accuracy.append(avg_mape_across_batches)

#training the model
if training : 
    for epoch in range(num_epochs) : 
        train_one_epoch()
        validate_one_epoch()
        

#Loss during training

fig_loss, ax_loss = plt.subplots()

ax_loss.plot(training_loss, label='Training Loss')
ax_loss.plot(validation_loss, label='Validation Loss')

ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.legend()

filename_loss = 'loss_plot.png'

save_image(fig_loss, filename_loss)


#MAPE during the training
fig_mape, ax_mape = plt.subplots()

ax_mape.plot(training_accuracy, label='Training MAPE')
ax_mape.plot(validation_accuracy, label='Validation MAPE')

ax_mape.set_xlabel('Epoch')
ax_mape.set_ylabel('MAPE')
ax_mape.legend()

filename_mape = 'mape_plot_evolution.png'

save_image(fig_mape, filename_mape)


def results(m) : 
    
    print(f"Résultats à t+{m}")       
    
    #We apply the model on the training set
    #We extract the right data at t+m
    #We define the reference sequence y_test
    #We calculate MAPE to find the error
    #We plot both courb to compare on the last 500 values ?
    #We plot the deviation courb

 
    #Results on the training set
    #Applying the model
    with torch.no_grad():
        training_predictions = model(X_train.to(device)).detach().cpu().numpy()
        #print("size training pred", type(training_predictions), training_predictions)
    #Extraction of data
    pred_values_train = training_predictions[m:,m-1].reshape(-1,1)
    real_values_train = y_train[:-m, m-1]
    
    #print("pred_values_train", pred_values_train.shape, pred_values_train)
    
    #Normalization
    pred_values_train = scaler.inverse_transform(pred_values_train)
    real_values_train = scaler.inverse_transform(real_values_train.to('cpu'))
    #Log back   
    pred_values_train = np.exp(pred_values_train) - 1
    real_values_train = np.exp(real_values_train) - 1
    
    #Calculation of MAPE
    mape_train = MAPE(real_values_train, pred_values_train)
    print(f"MAPE of the sensor {sensor} at t+{m} on training values =", mape_train)
    #Plot of the fitting on the last 200 values
    fig_res_train, ax_res_train = plt.subplots() 
    ax_res_train.plot(real_values_train[len(real_values_train)-200:], label='Real values training')
    ax_res_train.plot(pred_values_train[len(real_values_train)-200::], label='Training predictions')
    ax_res_train.set_xlabel('Time')
    ax_res_train.set_ylabel('Air pollution')
    #Saving it
    save_image(fig_res_train, f'Results_training_{sensor}_t+{m}.png')
    
    #Plot of the deviation courb
    plt.figure(figsize=(8, 6))
    plt.scatter(real_values_train, pred_values_train, color='blue', label='Real vs predicted values training')
    # Diagonal lign
    plt.plot(real_values_train, real_values_train, color='red', linestyle='--', label='Equality')
    plt.ylabel('Predicted values')
    plt.xlabel('Real values')

    plt.legend()
    plt.title(f'{sensor} at t+{m}')
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, f'{sensor}_at_t+{m}_plot.png'))
    plt.close()
    
    
    #Results on the testing set
    with torch.no_grad():
        testing_predictions = model(X_test.to(device)).detach().cpu().numpy()
    
    #Extraction of data
    pred_values_test = testing_predictions[m:,m-1].reshape(-1,1)
    real_values_test = y_test[:-m, m-1]
    
    #Normalization
    pred_values_test = scaler.inverse_transform(pred_values_test)
    real_values_test = scaler.inverse_transform(real_values_test.to('cpu'))
    #Log back   
    pred_values_test = np.exp(pred_values_test) - 1
    real_values_test = np.exp(real_values_test) - 1
    
    #Calculation of MAPE
    mape_test = MAPE(real_values_test, pred_values_test)
    print(f"MAPE of the sensor {sensor} at t+{m} on testing values =", mape_test)
    #Plot of the fitting
    fig_res_test, ax_res_test = plt.subplots() 
    ax_res_test.plot(real_values_test[len(real_values_test)-200:], label='Real values testing')
    ax_res_test.plot(pred_values_test[len(real_values_test)-200::], label='Testing predictions')
    ax_res_test.set_xlabel('Time')
    ax_res_test.set_ylabel('Air pollution')
    #Saving it
    save_image(fig_res_test, f'Results_testing_{sensor}_t+{m}.png')
    
    #Plot of the deviation courb
    plt.figure(figsize=(8, 6))
    plt.scatter(real_values_test, pred_values_test, color='blue', label='Real vs predicted values testing')
    # Diagonal lign
    plt.plot(real_values_test, real_values_test, color='red', linestyle='--', label='Equality')
    plt.ylabel('Predicted values')
    plt.xlabel('Real values')

    plt.legend()
    plt.title(f'{sensor} at t+{m}')
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, f'{sensor}_at_t+{m}_plot.png'))
    plt.close()
    
results(2)
results(5)
results(10)
results(24)

    
    
    



