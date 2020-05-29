"""Trainer for classifying MNIST"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ConvNet


def generate_datasets(mean, std):
    trn_data = datasets.MNIST(".",
                              train=True,
                              download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean, std)
                                  ])
                              )
    val_data = datasets.MNIST(".",
                              train=False,
                              download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean, std)
                                  ])
                              )
    return trn_data, val_data

def train(model, optim, trn_loader, val_loader, loss_fn, epochs=100, early_stopping=5, device='cpu'):
    model.to(device)
    loss_fn.to(device)

    # loss history
    tr_losses = []
    val_losses = []
    early_stopping_counter = 0

    for epoch in range(epochs):
        # Train one epoch
        model.train()
        tot_tr_loss = 0.
        for X, y in trn_loader:
            X = X.to(device)
            y = y.to(device)
            score = model(X)
            loss_tsr = loss_fn(score, y)
            
            optim.zero_grad()
            loss_tsr.backward()
            optim.step()

            loss = loss_tsr.detach().to('cpu').item()
            tot_tr_loss += loss
        tr_losses.append(tot_tr_loss)
        
        # Evaluate validation dataset
        model.eval()
        tot_val_loss = 0.
        correct = 0.
        total = 0.  # The size of valid datasets.
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)
            with torch.no_grad():
                score = model(X)
                _, pred = torch.max(score, dim=1)
                correct += (pred == y).sum() 
                total += len(y)
                loss_tsr = loss_fn(score, y)
                loss = loss_tsr.to('cpu').item()
            tot_val_loss += loss
        val_losses.append(tot_val_loss)
        accuracy = 100 * correct / total
        print(f"Epoch-{epoch}/{epochs} | tr_loss: {tot_tr_loss:.4f} | val_loss: {tot_val_loss:.4f} | accuracy: {accuracy:.2f}%")

        # If validation loss increases, count the early stopping counter.        
        # If not, reset the counter.
        if not val_losses or val_losses[-1] > loss:
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        # Criterion for early stopping.
        if early_stopping_counter >= early_stopping:
            break

    return {'tr_losses': tr_losses, 'val_losses': val_losses, 'accuracy': accuracy}


if __name__ == "__main__":
    # Configurations
    mean = (0.1307,)  # Mean of training data
    std = (0.3081,)  # Standard deviataion of training data 
    batch_size = 64
    learning_rate = 1e-3
    epochs=10
    early_stopping=2
    seed = 2020

    # Fix random seed
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Generate Datasets classes of training and validation
    trn_data, val_data = generate_datasets(mean, std)
    # Generate DataLoader classes
    trn_loader = DataLoader(trn_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConvNet()

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    train(model, optim, trn_loader, val_loader, loss_fn, epochs, early_stopping, device)
