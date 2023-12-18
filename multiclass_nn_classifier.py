# Modified version of nn_classifier.py for 40-class classification task

import torch, numpy as np, pickle
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from data_creator import load_train_test_data, get_ae_data, get_lda_data, get_pca_data
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from metrics import get_metrics_classicalml


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MulticlassClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MulticlassClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        x = torch.from_numpy(x)
        y = torch.from_numpy(np.array(y)).float()   # PyTorch expects ndarray. We could have reshaped it and sent it also
        return x, y



def model_train(input_size,hidden_size,batch_size, num_epochs, dim_reduction):


    print(f"Training nn with {dim_reduction}")
    data_folder= os.getcwd() + "/att_faces"
    train_data, train_label, val_data, val_label = load_train_test_data(data_folder)
    learning_rate = 1e-5
    model = MulticlassClassifier(input_size, hidden_size,40).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Generating data with {dim_reduction}")
    train_arr =None
    val_arr=None
    if(dim_reduction=="ae"):
        train_arr = get_ae_data(train_data)
        val_arr =   get_ae_data(val_data)
    elif(dim_reduction=="pca"):
        train_arr, test_data_arr = get_pca_data(train_data,val_data)
    elif(dim_reduction=="lda"):
        train_arr,test_data_arr = get_lda_data(train_data,val_data)


    print("Train Shape - ", train_arr.shape)
    print("Test Shape - ", train_arr.shape)
    train_dataset = CustomDataset(train_arr, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = CustomDataset(val_arr, val_label)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print("Start training...")
    train_losses, val_losses = [], []
    for epoch in tqdm(range(num_epochs)):

        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            targets = targets.unsqueeze(1) # From torch.Size([16]) to torch.Size([16,1])
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #train_loss += loss.item() * inputs.size(0)
            train_loss += loss.item()
        #train_loss /= len(train_loader.dataset)

        alpha = len(train_loader) // batch_size
        epoch_train_loss = train_loss / alpha
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                targets = targets.unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                valid_loss += loss.item() * inputs.size(0)
                valid_loss += loss.item()
            valid_loss /= len(valid_loader.dataset)
            alpha = len(valid_loader) // batch_size
            epoch_val_loss = valid_loss / alpha
            val_losses.append(epoch_val_loss)

        tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {round(epoch_train_loss,3)}, Valid Loss: {round(epoch_val_loss,3)}")

    print("Testing...")
    model.eval()
    for i in tqdm(range(0, len(val_arr), batch_size)):

        batch_data = val_arr[i:i+batch_size]

        outputs = model(inputs)
        pred_labels = torch.round(outputs)
        pred_list+=pred_labels.cpu().numpy().flatten().tolist()        
    pred_labels_arr = np.array(pred_list)
    print('pred shape',pred_labels_arr.shape)
    #print(classification_report(test_label, pred_labels_arr))
    cm = confusion_matrix(val_label, pred_labels_arr)
    ac,pr,re,f1 = get_metrics_classicalml(val_label, pred_labels_arr)
    print("Accuracy - ",round(ac,3))
    print("Precision - ",round(pr,3))
    print("Recall - ",round(re,3))
    print("F-1 - ",round(f1,3))

    print(classification_report(val_label,pred_labels_arr))
    print("Confusion Matrix:")
    print(cm)




if __name__=="__main__":


    model_train(768,50,16,200,"ae")
    model_train(768,50,16,200,"pca")
    model_train(768,50,16,200,"lda")