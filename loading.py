import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from widis_lstm_tools.preprocessing import random_dataset_split, inds_to_one_hot

from dataset import MolDataset
from model import MolModel
from train import train

# device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load dataset
filepath = r'data\smiles_train.txt'
data = MolDataset(filepath, 100)

# data splitting
train_data, test_data = random_dataset_split(data, split_sizes=(90 / 100., 10 / 100))

# data loader
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# model
model = MolModel(n_inputs=len(data.id2char), hidden_size=64).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)

# training
train(model, optimizer, criterion, 10, train_loader,
      test_loader, scheduler=scheduler)

model.cpu()


@torch.no_grad()
def transform(idx):
    # one_hot encoding
    idx = inds_to_one_hot(np.array(idx), len(data.id2char))

    # (batch_size, seq_len, input_size)
    idx = torch.Tensor(idx).reshape(1, 1, len(data.id2char))
    return idx


@torch.no_grad()
def generate_one_mol(x=0, k=2):
    x = transform(x)

    out, hiddens = model.forward(x)

    topk = torch.topk(out, k=k)

    topk_prob = torch.softmax(topk[0].squeeze(), dim=-1).numpy()

    topk_ids = topk[1].squeeze().numpy()

    pred = np.random.choice(topk_ids, p=topk_prob)

    inputs = transform(pred)

    molecule = [data.id2char[pred]]

    while True:
        out, hiddens = model.forward(inputs, hiddens)

        topk = torch.topk(out, k=k)

        topk_prob = torch.softmax(topk[0].squeeze(), dim=-1).numpy()

        topk_ids = topk[1].squeeze().numpy()

        out_id = np.random.choice(topk_ids, p=topk_prob)

        if out_id == 1:
            break

        inputs = transform(out_id)
        molecule.append(data.id2char[out_id])

    return ''.join(molecule)


def generate_molecules(n=10000, k=5):
    generated = []
    for _ in tqdm(range(n)):
        while True:
            molecule = generate_one_mol(n=n, k=k)

            valid_test = Chem.MolFromSmiles(molecule)

            if valid_test is not None and molecule not in generated and molecule not in data.list_molecules:
                print(len(generated))
                generated.append(molecule)
                break
    return generated


generated = generate_molecules(10000)

with open('results/sub.txt', 'a') as f:
    for mol in generated:
        f.write(mol)
        f.write('\n')