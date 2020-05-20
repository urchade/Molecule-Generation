import torch
from torch import nn
from torch.utils.data import DataLoader
from widis_lstm_tools.preprocessing import random_dataset_split, inds_to_one_hot

from dataset import MolDataset
from model import MolModel
import numpy as np

filepath = r'data\smiles_train.txt'

data = MolDataset(filepath, 100)

train_data, test_data = random_dataset_split(data, split_sizes=(90 / 100., 10 / 100))

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

model = MolModel(n_inputs=len(data.id2char), hidden_size=32)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)

for i in range(20):
    losses = []
    for x, y in train_loader:
        optimizer.zero_grad()
        y_hat, _ = model(x)
        loss = criterion(y_hat, y.long())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(np.mean(losses))


@torch.no_grad()
def transform(integer):
    integer = inds_to_one_hot(np.array(integer), len(data.id2char))
    return torch.Tensor(integer).reshape(1, 1, 40)


@torch.no_grad()
def pred_next(x=0):
    x = transform(x)

    out, hiddens = model.forward(x)

    pred = torch.argmax(out).item()

    inputs = transform(pred)

    molecule = [data.id2char[pred]]

    while True:
        out, hiddens = model.forward(inputs, hiddens)

        topk = torch.topk(out, 3)[1].squeeze().numpy()

        out_id = np.random.choice(topk)

        if out_id == 1:
            break

        inputs = transform(out_id)
        molecule.append(data.id2char[out_id])

    return molecule


from rdkit import Chem

mll = pred_next()
m = Chem.MolFromSmiles(''.join(mll))

