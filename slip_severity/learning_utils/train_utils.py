import torch

def train_epoch(model, data_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for sequences, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)
