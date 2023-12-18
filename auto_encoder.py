

import torch, json, pickle, time, numpy as np, sys, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from data_creator import load_train_test_data
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"


class Autoencoder(nn.Module):
    def __init__(self, n_inputs, encoding_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Linear(768, n_inputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        return x
    

def train_autoencoder(batch_size, num_epochs,n_inputs,encoding_dim, save_interval=1,model_shorthand="gc",task='recognition'):
    print(f'Training auto-encoder for {task} task.')

    data_folder= os.getcwd() + "/att_faces"

    train_data, train_labels, test_data, test_labels= load_train_test_data(data_folder,task=task)

    print('Training data shape', train_data.shape)
    print('Testing data shape', test_data.shape)


    device = "cuda" if torch.cuda.is_available() else "cpu"



    train_dataset = CustomDataset(train_data)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if test_data is not None:
        valid_dataset = CustomDataset(test_data)
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    autoencoder = Autoencoder(n_inputs, encoding_dim)
    autoencoder = autoencoder.to(device)  # Move model to GPU
    criterion = nn.MSELoss()

    optimizer = optim.AdamW(autoencoder.parameters(), lr=1e-4, weight_decay=1e-8)

    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training scaler
    train_losses, val_losses = [], []

    print("Start the training...")
    for epoch in tqdm(range(num_epochs)):

        start_time = time.time()
        train_loss = 0.0
        autoencoder.train()
        for _, batch_data in enumerate(train_data_loader):

            train_input = torch.tensor(batch_data).to(torch.float32).to(device)
            optimizer.zero_grad()
            outputs = autoencoder(train_input)
            loss = criterion(outputs, train_input)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        alpha = len(train_data_loader) // batch_size
        epoch_train_loss = train_loss / alpha
        train_losses.append(epoch_train_loss)
    

        if test_data is not None:
            autoencoder.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for _, val_batch_data in enumerate(valid_data_loader):
                    val_input = torch.tensor(val_batch_data).to(torch.float32).to(device)
                    val_outputs = autoencoder(val_input)
                    val_loss = criterion(val_outputs,val_input)                    

                    val_loss_sum+=val_loss.item()

            alpha = len(valid_data_loader) // batch_size
            epoch_val_loss = val_loss_sum / alpha
            val_losses.append(epoch_val_loss)


        print(f'Epoch {epoch+1} \t\t Training Loss: {epoch_train_loss} \t\t Validation Loss: {epoch_val_loss}')
        
        print(f"Time for epoch {epoch+1} - ", time.time()-start_time)

    plt.figure(figsize=(10,6))
    plt.plot(train_losses,label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(10,6))
    plt.plot(val_losses,label='training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Save Model
    checkpoint_folder_path = "./trained_models"
    os.makedirs(checkpoint_folder_path, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder_path,f"autoencoder_{model_shorthand}_pn_{encoding_dim}_{task}.pth")
    torch.save(autoencoder.state_dict(), checkpoint_path)

    print(f"Training losses saved. Train Loss={len(train_losses)}, Validation Loss={len(val_losses)}")

    return autoencoder


def load_autoencoder_model(model_path, n_inputs, encoding_dim, device):
    autoencoder = Autoencoder(n_inputs, encoding_dim)
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.to(device)
    autoencoder.eval()
    return autoencoder

def get_bottleneck_representation(em, input_dim, encoding_dim,task):

    model_path = f"./trained_models/autoencoder_gc_pn_256_{task}.pth"  # Path to the saved model
    n_inputs = input_dim  # Input dimension
    encoding_dim = encoding_dim  # Latent space dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em=torch.tensor(em).to(torch.float32).to(device=device)
    autoencoder = load_autoencoder_model(model_path, n_inputs, encoding_dim, device)
    with torch.no_grad():
        bottleneck_representation = autoencoder.encoder(em)
    return bottleneck_representation

def get_ae_data(data,task):
    print('Generating data using autoencoder')
    return get_bottleneck_representation(data,10304,256,task=task)

if __name__=="__main__":

    print("Start")
    train_autoencoder(8,30,10304,256,save_interval=30,model_shorthand="gc",task='recognition')
    train_autoencoder(8,30,10304,256,save_interval=30,model_shorthand="gc",task='identification')
    print('Finished training auto-encoder models')

