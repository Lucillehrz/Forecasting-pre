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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

nb_sensors = 200
path = f'/scratch_dgxl/lhuriez/forecasting/data/data_sensors_activated/data_200_all_activated.csv'

output_dir = f'/scratch_dgxl/lhuriez/forecasting/Results/all_sensors_t+n'
print("data path:", path)

data_tot = pd.read_csv(path, parse_dates=[0]) 
data = pd.DataFrame(data_tot) 
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)
df = dc(data)
df_array = df.values

#Logarithme et normalisation
df_array = np.log(df_array + 1)
scaler = MinMaxScaler(feature_range=(0, 1))
df_array = scaler.fit_transform(df_array)

#Parameters
sequence_length = 5

hidden_size = 5
num_layers = 4
learning_rate = 0.0001
num_epochs = 50
input_size = nb_sensors

n = 24

batch_size = 32

training = True
saving = False

print('sequence_length =', sequence_length)
print('input_size =', input_size)
#print('output_size =', output_size)
print('hidden_size =', hidden_size)
print('num_layers =', num_layers)
print('learning_rate =', learning_rate)
print('num_epochs =', num_epochs)
print('batch_size =', batch_size)

#Creation and saving figure
def save_image(fig, filename):
    file_path = os.path.join(output_dir, filename)
    fig.savefig(file_path)
    plt.close(fig)
    print(f'Image saved at {file_path}')
    
def MAPE(y_true, y_pred):
    

    if len(y_true) != len(y_pred):
        raise ValueError("Les séries doivent avoir la même longueur.")

    mask = y_true != 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    # Calcul de la MAPE sans les valeurs où y_true est 0
    mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    
    return mape

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
    
#Définition du modèle
class MultiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_steps, dropout=0.5):
        super(MultiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        self.n_steps = n_steps
        self.output_size = input_size*n_steps

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=self.bidirectional, dropout=dropout)
        
        self.fc = nn.Linear(hidden_size * 2, self.output_size) if self.bidirectional else nn.Linear(hidden_size, output_size)
        

    def forward(self, x):
        batch_size = x.size(0)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = out.reshape([batch_size, self.n_steps, -1])

        return out
    
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


model =  MultiLSTM(input_size, hidden_size, num_layers, n).to(device)
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
        #print("shape")
        #print("x",x_batch.shape, "out", output.shape, "y", y_batch.shape)
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
        
with torch.no_grad():
    training_predictions = model(X_train.to(device)).to('cpu').numpy()
    test_predictions = model(X_test.to(device)).detach().cpu().numpy()
    
def results(m, i) : 
    
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
    pred_values_train = training_predictions[m:,m-1,:]
    real_values_train = y_train[:-m, m-1]
    
    print("pred_values_train", pred_values_train.shape)
    
    #Normalization
    pred_values_train = scaler.inverse_transform(pred_values_train.reshape(-1, nb_sensors))
    real_values_train = scaler.inverse_transform(real_values_train.to('cpu'))
    #Log back   
    pred_values_train = np.exp(pred_values_train) - 1
    real_values_train = np.exp(real_values_train) - 1
    
    #Calculation of MAPE
    mape_train = MAPE(real_values_train, pred_values_train)
    print(f"MAPE of all sensors at t+{m} on training values =", mape_train)
    
    #Example of a sensor, index i
    #Plot of the fitting
    fig_res_train, ax_res_train = plt.subplots() 
    ax_res_train.plot(real_values_train[len(real_values_train)-200:,i], label='Real values training')
    ax_res_train.plot(pred_values_train[len(real_values_train)-200::,i], label='Training predictions')
    ax_res_train.set_xlabel('Time')
    ax_res_train.set_ylabel('Air pollution')
    #Saving it
    save_image(fig_res_train, f'Results_training_t+{m}.png')
    
    #Plot of the deviation courb
    plt.figure(figsize=(8, 6))
    plt.scatter(real_values_train[:,i], pred_values_train[:,i], color='blue', label='Real vs predicted values training')
    # Diagonal lign
    plt.plot(real_values_train[:,i], real_values_train[:,i], color='red', linestyle='--', label='Equality')
    plt.ylabel('Predicted values')
    plt.xlabel('Real values')

    plt.legend()
    plt.title(f' at t+{m}')
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, f'_at_t+{m}_plot.png'))
    plt.close()
    
    
    #Results on the testing set
    with torch.no_grad():
        testing_predictions = model(X_test.to(device)).detach().cpu().numpy()
    
    #Extraction of data
    pred_values_test = testing_predictions[m:,m-1]
    real_values_test = y_test[:-m, m-1]
    
    #Normalization
    pred_values_test = scaler.inverse_transform(pred_values_test.reshape(-1, nb_sensors))
    real_values_test = scaler.inverse_transform(real_values_test.to('cpu'))
    #Log back   
    pred_values_test = np.exp(pred_values_test) - 1
    real_values_test = np.exp(real_values_test) - 1
    
    #Calculation of MAPE
    mape_test = MAPE(real_values_test, pred_values_test)
    print(f"MAPE of all sensors at t+{m} on testing values =", mape_test)
    
    #Calculation of MAPE for one sensor
    mape_test_i = MAPE(real_values_test[:,i], pred_values_test[:,i])
    print(f'MAPE of the {i}th sensor at t+{m} on testing values =', mape_test_i)
    
    #Plot of the fitting
    fig_res_test, ax_res_test = plt.subplots() 
    ax_res_test.plot(real_values_test[len(real_values_test)-200:, i], label='Real values testing')
    ax_res_test.plot(pred_values_test[len(real_values_test)-200::, i], label='Testing predictions')
    ax_res_test.set_xlabel('Time')
    ax_res_test.set_ylabel('Air pollution')
    #Saving it
    save_image(fig_res_test, f'Results_testing_t+{m}_{i}.png')
    
    #Plot of the deviation courb
    plt.figure(figsize=(8, 6))
    plt.scatter(real_values_test[:,i], pred_values_test[:,i], color='blue', label='Real vs predicted values testing')
    # Diagonal lign
    plt.plot(real_values_test[:,i], real_values_test[:,i], color='red', linestyle='--', label='Equality')
    plt.ylabel('Predicted values')
    plt.xlabel('Real values')

    plt.legend()
    plt.title(f'at t+{m}')
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, f'at_t+{m}_plot_{i}.png'))
    plt.close()
    
    MAPE_list_train = []
    MAPE_list_test = []
    Numero = []
    too_high = []

    for i in range(nb_sensors): 
        m_train = MAPE(real_values_train[:,i], pred_values_train[:,i])
        m_test = MAPE(real_values_test[:-1,i], pred_values_test[1:,i])
        if m_train<400 and m_test<400 : 
            MAPE_list_train.append(m_train)
            MAPE_list_test.append(m_test)
            Numero.append(i)
        else : 
            too_high.append([i, m_train, m_test])
    
    print("Some values of MAPE were too high : ", too_high)

    plt.figure()    
    plt.bar(Numero, MAPE_list_train, label='Training')
    plt.bar(Numero, MAPE_list_test, label='Testing', alpha=0.5)
    plt.xlabel('Sensors')
    plt.ylabel('MAPE') 
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'hist_at_t+{m}_plot.png'))
    

    
    
i=99
results(2, i)
if n >= 5 : 
    results(5, i)
    
if n >= 10 : 
    results(10, i)
    
if n >= 24 : 
    results(24, i)
