---
description: torch를 이용한 간단한 Regressoin Model을 구현해봅시다.
---

# 😀 Regression

{% code lineNumbers="true" %}
```python
import os

import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.5f}'.format

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader

# Device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Data Load
califonia = fetch_california_housing()
df = pd.DataFrame(califonia.data, columns=califonia.feature_names)
df['target'] = califonia.target

# Dataset Class
class MyData(Dataset):
    def __init__(self, df):
        self.df = df
        self.x = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1:].values
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        # index로 row 하나를 특정합니다.
        x = self.x[idx] 
        y = self.y[idx] 
        return torch.tensor(x, dtype = torch.float), torch.tensor(y, dtype = torch.float)

idx_pivot = 14000
batch_size = 256

# Train, Valid Split
train = df[:idx_pivot].reset_index(drop = True)
valid = df[idx_pivot:].reset_index(drop = True)

# Dataset
train_ds = MyData(df = train)
valid_ds = MyData(df = valid)

# train_loader, valid_loader
train_loader = DataLoader(train_ds, shuffle = True, batch_size = batch_size)
valid_loader = DataLoader(valid_ds, shuffle = False, batch_size = batch_size)

class RegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        y = self.fc2(x)
        return y

model = RegressionModel(8,1).to(device)
optim = torch.optim.Adam(model.parameters(), lr = 1e-3)
loss_fn = nn.MSELoss().to(device)

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train() # model을 학습모드로 변경
    
    train_loss = 0
    for data in dataloader:
        x = data[0].to(device)
        y = data[1].to(device)
        y_pred = model(x)
        
        loss = loss_fn(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    train_loss /= len(dataloader) # epoch 당 train loss 평균값
    
    return train_loss

@torch.no_grad()
def valid_one_epoch(model, dataloader, loss_fn, device):
    model.eval() # model을 평가모드로 변경
    
    valid_loss = 0

    with torch.no_grad():
        for data in dataloader:
            x = data[0].to(device)
            y = data[1].to(device)
            y_pred = model(x)

            loss = loss_fn(y_pred, y)

            valid_loss += loss.item()

    valid_loss /= len(dataloader)
    return valid_loss

def run_train(model = model, n_epochs = 150,
             train_loader = train_loader, valid_loader = valid_loader,
             loss_fn = loss_fn, optimizer = optim, device = device):
    
    train_hs, valid_hs = [], []
    print_iter= 20
    lowest_loss, lowest_epoch = np.inf, np.inf
    early_stop = 20
    
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model = model, 
                                       dataloader = train_loader,
                                       loss_fn = loss_fn, 
                                       optimizer = optimizer,
                                       device = device)
        
        valid_loss = valid_one_epoch(model = model, 
                                       dataloader = valid_loader,
                                       loss_fn = loss_fn, 
                                       device = device)
        
        train_hs.append(train_loss)
        valid_hs.append(valid_loss)
        
        # monitoring
        if (epoch + 1) % print_iter == 0:
            print(f"Epoch {epoch+1} | Train Loss : {train_loss:.3e} | Val Loss : {valid_loss:.3e} | Lowest Loss : {lowest_loss:.3e} |")
           
        # Lowest Loss 갱신  - Valid Loss 기준
        if valid_loss < lowest_loss:
            lowest_loss = valid_loss
            lowest_epoch= epoch
            # model 저장
            torch.save(model.state_dict(), './model_regression.bin') # pth, pt
        else:
            if early_stop > 0 and lowest_epoch+ early_stop < epoch +1:
                print("print("Early Stopping..!")") 
                break
                
    print()            
    print(f"The Best Validation Loss={lowest_loss:.3e} at {lowest_epoch} Epoch")
    
    result = dict()
    result["Train Loss"] = train_hs
    result["Valid Loss"] = valid_hs
    
    return model, result


model, result = run_train()

def plot(res):
    plot_from = 0
    plt.figure(figsize=(20, 10))
    plt.title("Train/Valid Loss History", fontsize = 20)
    plt.plot(
        range(0, len(res['Train Loss'][plot_from:])), 
        res['Train Loss'][plot_from:], 
        label = 'Train Loss'
        )

    plt.plot(
        range(0, len(res['Valid Loss'][plot_from:])), 
        res['Valid Loss'][plot_from:], 
        label = 'Valid Loss'
        )

    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()

plot(result)
```
{% endcode %}

<figure><img src="../.gitbook/assets/image (17).png" alt=""><figcaption></figcaption></figure>
