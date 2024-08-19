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

nb_sensors = 100
#path = f'/scratch_dgxl/lhuriez/forecasting/data/data_sensors_activated/data_100_all_activated.csv'
#path = f'/scratch_dgxl/lhuriez/forecasting/data/data_sensors_activated/data_200_all_activated.csv'
path = r"C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\Forecast-air-pollution\data\data_sensors_activated\data_300_all_activated.csv"
output_dir = r'C:\Users\Lucille\OneDrive\Documents\Forecasting air pollution\forecasting\forecasting\Results\all_sensors_t+1'

print("data path:", path)


data_tot = pd.read_csv(path, parse_dates=[0]) 
data = pd.DataFrame(data_tot) 
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)
df = dc(data)
df_array = df.values

sequence_length = 1

hidden_size = 10
num_layers = 2
learning_rate = 0.0001
num_epochs = 50

batch_size = 32

training = True

input_size = nb_sensors 
output_size = nb_sensors 

print('sequence_length =', sequence_length)
print('hidden_size =', hidden_size)
print('num_layers =', num_layers)
print('learning_rate =', learning_rate)
print('num_epochs =', num_epochs)
print('batch_size =', batch_size)
print('n =', n)

#Logarithme et normalisation
df_array = np.log(df_array + 1)
scaler = MinMaxScaler(feature_range=(0, 1))
df_array = scaler.fit_transform(df_array)

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

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
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
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(MultiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=self.bidirectional, dropout=dropout)
        
        self.fc = nn.Linear(hidden_size * 2, output_size) if self.bidirectional else nn.Linear(hidden_size, output_size)
        

    def forward(self, x):
        batch_size = x.size(0)
        out, _ = self.lstm(x)
        
        out = self.fc(out[:, -1, :])
        return out

        
# Créer les séquences de données
X, y = create_sequences(df_array, sequence_length)

#Séparation données training and test
split_index = int(len(X)*0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# Convertir les données en tensors PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#Initialisation du modèle
model = MultiLSTM(input_size, hidden_size, num_layers, output_size).to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#List in whihc we will store accuracy and loss of the training and validation
training_loss = []
training_accuracy = []

validation_loss = []
validation_accuracy = []


def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    train_loss = 0.0
    train_mape = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        output = model(x_batch)
        loss = loss_function(output, y_batch)

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
    
    
    training_loss.append(train_loss)
    training_accuracy.append(train_mape)
    
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_mape:.2f}%')
   
    
def validate_one_epoch():
    model.train(False)
    running_loss = 0.0
    running_mape = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
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
    
if training : 
    for epoch in range(num_epochs):
        train_one_epoch()
        validate_one_epoch()
        
with torch.no_grad():
    training_predictions = model(X_train.to(device)).to('cpu').numpy()
    test_predictions = model(X_test.to(device)).detach().cpu().numpy()
    
training_predictions = scaler.inverse_transform(training_predictions.reshape(-1,output_size))
new_y_train = scaler.inverse_transform(y_train.to('cpu').numpy())
training_predictions = np.exp(training_predictions) - 1
new_y_train = np.exp(new_y_train) - 1

test_predictions = scaler.inverse_transform(test_predictions.reshape(-1,output_size))
new_y_test = scaler.inverse_transform(y_test.to('cpu').numpy())
test_predictions = np.exp(test_predictions) - 1
new_y_test = np.exp(new_y_test) - 1

pred_train = training_predictions[1:,:]
real_train = new_y_train[:-1,:]

pred_test = test_predictions[1:,:]
real_test = new_y_test[:-1,:]

#MAPE tot
mape_train = MAPE(pred_train, real_train)
mape_test = MAPE(pred_test, real_test)

print("MAPE on the training set: ", mape_train)
print("MAPE on the testing set: ", mape_test)

#Graph loss training
#Creating axes and courbs
fig_loss, ax_loss = plt.subplots()
# Tracer les courbes sur les axes
ax_loss.plot(training_loss, label='Training Loss')
ax_loss.plot(validation_loss, label='Validation Loss')
# Ajouter les labels et la légende
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.legend()
# Nom du fichier pour l'image
filename_loss = 'loss_plot.png'
# Enregistrer l'image
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


def focus(i) : 
    real_test_i = real_test[:,i]
    pred_test_i = pred_test[:,i]
    
    #Dispersion graph
    # Dispersion diagram on the testing set
    plt.figure(figsize=(8, 6))
    plt.scatter(real_test_i, pred_test_i, color='blue', label='Real vs predicted values')
    # Diagonal lign
    plt.plot(real_test_i, real_test_i, color='red', linestyle='--', label='Equality')
    plt.ylabel('Predicted values')
    plt.xlabel('Real values')

    plt.legend()
    plt.title('Dispersion diagram on the testing set')
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, f'all_test_scatter_plot.png'))
    plt.close()
    
    #MAPE
    mape_test = MAPE(real_test_i, pred_test_i)
    print(f"MAPE of the sensor at t+1 on the testing values =", mape_test)
    
    #Plot of the fitting
    fig_res_test, ax_res_test = plt.subplots() 
    ax_res_test.plot(real_test_i[len(real_test_i)-200::], label='Real values')
    ax_res_test.plot(pred_test_i[len(real_test_i)-200::], label='Predictions')
    ax_res_test.set_xlabel('Time')
    ax_res_test.set_ylabel('Air pollution')
    #Saving it
    save_image(fig_res_test, f'Results_testing_one_plot.png')
    
    
    
focus(2)
focus(57)
