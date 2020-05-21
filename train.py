import numpy as np
import torch
from tqdm import tqdm


def evaluate(model, data, criterion):
    model.eval()
    losses = []
    device = next(model.parameters()).device

    for x, y in tqdm(data, leave=False):
        y_hat, _ = model(x.to(device))
        loss = criterion(y_hat.transpose(1, 2), y.long().to(device))
        losses.append(loss.item())

    return np.mean(losses)


def update(model, data, criterion,
           optimizer):
    losses = []
    model.train()
    device = next(model.parameters()).device

    for x, y in tqdm(data, leave=False):
        optimizer.zero_grad()
        y_hat, _ = model.forward(x.to(device))
        loss = criterion(y_hat.transpose(1, 2), y.long().to(device))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return np.mean(losses)


def train(model, optimizer, criterion, epoch, train_data, test_data,
          scheduler=None):
    train_losses = []
    valid_losses = []

    best_model = np.infty

    with tqdm(range(epoch), desc=f"Train acc = {0: .4f} || Val acc = {0: .4f}") as pbar:
        for _ in pbar:
            train_loss = update(model=model, data=train_data, criterion=criterion,
                                optimizer=optimizer)

            valid_loss = evaluate(model=model, data=test_data, criterion=criterion)

            if scheduler:
                scheduler.step(valid_loss)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if valid_loss < best_model:
                best_model = valid_loss
                torch.save(model.state_dict(), 'model.pt')

            pbar.set_description(f"Train loss = {train_loss: .4f} || Val loss = {valid_loss: .4f}")

    return train_loss, valid_loss
