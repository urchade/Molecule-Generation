from torch.utils.data import Dataset

path = r'data\smiles_train.txt'


class MolDataset(Dataset):
    def __init__(self, filepath):

        list_molecules = []

        with open(filepath, 'r') as file:
            for mol in file:
                list_molecules.append(mol.strip('\n'))

        self.len_data = len(list_molecules)
        all_combined = ''.join(list_molecules)

        # 0 for padding, 1 for <sos>, 2 for <eos>
        self.id2char = dict(enumerate(set(all_combined), start=2))
        self.id2char[0] = '<sos>'
        self.id2char[1] = '<eos>'

        self.char2id = {v: k for k, v in self.id2char.items()}

        self.id_molecules = [[self.char2id[char] for char in molecule] for molecule in list_molecules]

        for i in range(self.len_data):
            self.id_molecules[i].append(1)  # Insert <eos> at the end
            self.id_molecules[i].insert(0, 0)  # Insert <sos> at the start

    def __len__(self):
        return self.len_data

    def __getitem__(self, item):
        pass
