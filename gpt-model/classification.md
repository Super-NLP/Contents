---
description: torch를 이용한 간단한 classification 모델을 구현해봅시다.
---

# 😃 Classification

{% code lineNumbers="true" %}
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms, datasets

# Colab
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Data Download
train_data = datasets.CIFAR10('./data', download= True, train = True, transform = transforms.ToTensor())

class MyDataset(Dataset):
    def __init__(self, data = train_data.data, label = train_data.targets):
        self.x = torch.tensor( data / 255. , dtype = torch.float).permute(0, 3, 1, 2)
        self.y = torch.tensor(label, dtype = torch.long) 

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def prepare_loaders(datas = train_data.data, labels = train_data.targets, index_num = 30000, batch = 128*2):
  
  train = datas[:index_num]
  valid = datas[index_num:]
  
  train_label = labels[:index_num] # 30000 labels
  valid_label = labels[index_num:] # 20000 labels
  
  train_ds = MyDataset(data = train, label = train_label)
  valid_ds = MyDataset(data = valid, label = valid_label)
  
  train_loader = DataLoader(train_ds, batch_size = batch, shuffle = True)
  valid_loader = DataLoader(valid_ds, batch_size = batch, shuffle = False)
  
  return train_loader, valid_loader
  
train_loader, valid_loader = prepare_loaders()

data = next(iter(train_loader))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 7, stride = 1, padding = 0)
        self.pool = nn.MaxPool2d(2, 2)
        k = 32*13*13
        self.seq = nn.Sequential(
            nn.Linear(k, 512), 
            nn.ReLU(), 
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=-1),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        y = torch.flatten(x, start_dim = 1) 
        y = self.seq(y)
        return y 

model = Model().to(device)
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())

def train_one_epoch(model = model, dataloader = train_loader, loss_fn = loss_fn, optimizer = optimizer, device = device, epoch = 1):
    model.train()
    
    train_loss, dataset_size = 0,  0
    preds, trues = [], []
    bar = tqdm(dataloader, total= len(dataloader))
    
    for data in bar:
        x = data[0].to(device)
        bs = x.shape[0]
        y_true = data[1].to(device)
        y_pred = model(x)
        
        loss = loss_fn(y_pred, y_true)
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        dataset_size += bs                           # 실시간으로 크기가 update
        train_loss += (loss.item() * bs)             # batch 단위 loss가 누적
        train_epoch_loss = train_loss / dataset_size # 실시간 train_loss
        
        preds.append(y_pred)
        trues.append(y_true)
        
        preds_cat = torch.cat(preds, dim = 0)
        trues_cat = torch.cat(trues, dim = 0)
        
        train_acc = 100 * (trues_cat == torch.argmax(preds_cat, dim=1)).sum().detach().cpu().item() / dataset_size # 전체 맞은 개수
        
        bar.set_description(f"Epoch : [{epoch:02d}] | Train Loss : [{train_epoch_loss:.3e}] | Accuracy : [{train_acc:.2f}]")
        
    return train_epoch_loss, train_acc

@torch.no_grad()
def valid_one_epoch(model = model, dataloader = valid_loader, loss_fn = loss_fn, device = device, epoch = 1):
    model.eval()
    
    valid_loss, dataset_size = 0,  0
    preds, trues = [], []
    bar = tqdm(dataloader, total= len(dataloader))
    
    with torch.no_grad():
        for data in bar:
            x = data[0].to(device)         # x: [bs, 3, 32, 32] 
            bs = x.shape[0]
            y_true = data[1].to(device)    # y_true: [bs]
            y_pred = model(x)              # y_pred: [bs, 10]

            loss = loss_fn(y_pred, y_true)

            dataset_size += bs                           # 실시간으로 크기가 update
            valid_loss += (loss.item() * bs)             # batch 단위 loss가 누적
            valid_epoch_loss = valid_loss / dataset_size # 실시간 train_loss

            preds.append(y_pred)  # preds: [ [bs, 10], [bs, 10], ...]
            trues.append(y_true)  # trues: [ [bs], [bs], [bs],  ...]

            preds_cat = torch.cat(preds, dim = 0) # preds_cat: [total_bs, 10]
            trues_cat = torch.cat(trues, dim = 0) # trues_cat: [total_bs]

            val_acc = 100 * (trues_cat == torch.argmax(preds_cat, dim=1)).sum().detach().cpu().item() / dataset_size # 전체 맞은 개수

            bar.set_description(f"Epoch : [{epoch:02d}] | Val Loss : [{valid_epoch_loss:.3e}] | Accuracy : [{val_acc:.2f}]")
        
    return valid_epoch_loss, val_acc

def run_train(model = model, train_loader = train_loader, valid_loader = valid_loader, 
              loss_fn = loss_fn, optimizer = optimizer, device = device, n_epochs = 150):
    
    train_hs, valid_hs, train_accs, valid_accs = [], [], [], []
    
    print_iter= 10
    
    lowest_loss, lowest_epoch = np.inf, np.inf
    
    early_stop = 20
    
    for epoch in range(n_epochs):
        
        train_loss, train_acc = train_one_epoch(model = model, optimizer = optimizer,
                                                dataloader = train_loader, 
                                                loss_fn = loss_fn,
                                                device = device, epoch = epoch)
        
        valid_loss, valid_acc = valid_one_epoch(model = model, 
                                                dataloader = valid_loader, 
                                                loss_fn = loss_fn,
                                                device = device, epoch = epoch)

        train_hs.append(train_loss)
        train_accs.append(train_acc)
        
        valid_hs.append(valid_loss)
        valid_accs.append(valid_acc)
        
        # 모니터링 
        if (epoch + 1) % print_iter == 0:
            print(f"Ep:[{epoch}]|TL:{train_loss:.3e}|VL:{valid_loss:.3e}|LL:{lowest_loss:.3e}")
            
        # Lowest Loss 갱신 -> valid loss 기준
        if valid_loss < lowest_loss:
            lowest_loss = valid_loss
            lowest_epoch = epoch
            torch.save(model.state_dict(), './model_classification.bin')
        else:
            if early_stop > 0 and lowest_epoch + early_stop < epoch +1:
                print("Early Stopping..!")
                break
                
    print()
    print("The Best Validation Loss= %.3e at %d Epoch" % (lowest_loss, lowest_epoch))
    
    # model load
    model.load_state_dict(torch.load('./model_classification.bin'))
    
    # result
    result = dict()
    result["Train Loss"] = train_hs
    result["Valid Loss"] = valid_hs
    
    result["Train Accs"] = train_accs
    result["Valid Accs"] = valid_accs
    
    return model, result

model, result = run_train()

def plot_loss(res):
    ## Train/Valid Loss History Visualization
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

def plot_acc(res):
    # Train/Valid Accuracy History Visualization
    plot_from = 0
    plt.figure(figsize=(20, 10))
    plt.title("Train/Valid Accuracy History", fontsize = 20)
    plt.plot(
        range(0, len(res['Train Accs'])), 
        res['Train Accs'], 
        label = 'Train Accs'
        )

    plt.plot(
        range(0, len(res['Valid Accs'])), 
        res['Valid Accs'], 
        label = 'Valid Accs'
        )

    plt.legend()
    # plt.yscale('log')
    plt.grid(True)
    plt.show()

plot_loss(result)
plot_acc(result)
```
{% endcode %}

<figure><img src="../.gitbook/assets/image (19).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../.gitbook/assets/image (20).png" alt=""><figcaption></figcaption></figure>
