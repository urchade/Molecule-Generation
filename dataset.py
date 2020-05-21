import numpy as np
from torch.utils.data import Dataset
from widis_lstm_tools.preprocessing import inds_to_one_hot


class MolDataset(Dataset):
    def __init__(self, filepath, seq_len):

        self.seq_len = seq_len

        self.list_molecules = []

        with open(filepath, 'r') as file:
            for mol in file:
                self.list_molecules.append(mol.strip('\n'))

        self.len_data = len(self.list_molecules)
        all_combined = ''.join(self.list_molecules)

        # 0 for padding, 1 for <sos>, 2 for <eos>
        self.id2char = dict(enumerate(set(all_combined), start=2))
        self.id2char[0] = '<sos>'
        self.id2char[1] = '<eos>'

        self.char2id = {v: k for k, v in self.id2char.items()}

        self.id_molecules = [[self.char2id[char] for char in molecule] for molecule in self.list_molecules]

        for i in range(self.len_data):
            self.id_molecules[i].append(1)  # Insert <eos> at the end
            self.id_molecules[i].insert(0, 0)  # Insert <sos> at the start

        self.combined = np.array([char for mols in self.id_molecules for char in mols])

    def __len__(self):
        return self.len_data // self.seq_len

    def __getitem__(self, item):

        x = inds_to_one_hot(self.combined[item * self.seq_len: item * self.seq_len + self.seq_len],
                            len(self.id2char))

        y = self.combined[item * self.seq_len + 1: item * self.seq_len + self.seq_len + 1]

        return x, y
